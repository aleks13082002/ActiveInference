import os
import math
import time
import keyboard
import wandb
import torch
import random as ran
from typing import Dict, Optional


wandb.login(key="YOUR_WANDB_API_KEY_HERE")
wandb.init(project="POMDP_Drone_Maintenance", entity="your_wandb_entity")


# =========================
# SENSOR
# =========================
class sensor:
    def __init__(self):
        self.temperature_sensor_activation = False
        self.anomaly_type = None

        self.mean_vals = torch.tensor([0.025, 55.0])
        self.std_vals = torch.tensor([0.01, 2.0])

        # perturbation
        self.perturbation_duration = 0
        self.instant_anomaly_mean = torch.tensor([0.1, 55.0])
        self.instant_anomaly_std = torch.tensor([0.02, 2.0])

        # overheating (persistent)
        self.persistent_anomaly_mean = torch.tensor([0.2, 75.0])
        self.persistent_anomaly_std = torch.tensor([0.01, 3.0])
        self.persistent_duration = False

    def anomalyInjection(self, anomaly_type):
        if anomaly_type == "perturbation":
            self.anomaly_type = anomaly_type
            self.perturbation_duration = ran.randint(10, 20)
            tmp = ran.uniform(0.09, 0.15)
            self.instant_anomaly_mean = torch.tensor([tmp, 50.0])

        elif anomaly_type == "overheating":
            self.anomaly_type = anomaly_type
            self.persistent_duration = True
            self.persistent_anomaly_mean = torch.tensor([0.1, 75.0])

        else:
            self.anomaly_type = None

    def get_data(self):
        """
        - Se temperature_sensor_activation è False → ritorna SOLO vibrazione (shape [1])
        - Se True → ritorna vibrazione+temperatura (shape [2])
        """

        # Persistent overheating
        if self.persistent_duration:
            print("Persistent anomaly ongoing due to overheating")
            if ran.random() < 0.2:
                self.persistent_anomaly_mean[1] += 1.0

            full_vec = torch.normal(self.persistent_anomaly_mean, self.persistent_anomaly_std)
            full_vec[0] = torch.clamp(full_vec[0], min=0.010)
            return full_vec if self.temperature_sensor_activation else full_vec[0:1]

        # Perturbation
        if self.perturbation_duration > 0:
            print("Perturbation anomaly ongoing " + str(self.perturbation_duration))

            if self.perturbation_duration == 1:
                self.anomaly_type = None
                self.perturbation_duration -= 1
            else:
                full_vec = torch.normal(self.instant_anomaly_mean, self.instant_anomaly_std)
                full_vec[0] = torch.clamp(full_vec[0], min=0.1)
                self.perturbation_duration -= 1
                return full_vec if self.temperature_sensor_activation else full_vec[0:1]

        # Normal
        if self.temperature_sensor_activation:
            full_vec = torch.normal(self.mean_vals, self.std_vals)
            full_vec[0] = torch.clamp(full_vec[0], min=0.010)
            return full_vec
        else:
            vib = torch.normal(self.mean_vals[0], self.std_vals[0]).unsqueeze(0)
            vib[0] = torch.clamp(vib[0], min=0.010)
            return vib


# =========================
# BATTERY (rimane nel mondo ma non nell'agente)
# =========================
class Battery:
    def __init__(self, capacity=100.0):
        self.capacity = float(capacity)
        self.level = float(capacity)

    def consume(self, amount):
        self.level = max(0.0, self.level - float(amount))

    def is_empty(self):
        return self.level <= 0.0

    def percentage(self):
        return (self.level / self.capacity) * 100.0


# =========================
# AGENT
# =========================
class agent:
    def __init__(self, battery: Battery, sensor_ref: sensor):
        self.battery = battery
        self.sensor = sensor_ref

        self.states = ["normal", "perturbation", "overheating"]
        self.n_states = len(self.states)

        self.belief = torch.ones(self.n_states) / self.n_states
        self.last_action = 0

        # --- energia (dinamica del mondo) ---
        self.cost_read_vibration = 0.04
        self.cost_activate_temp_sensor = 0.01
        self.cost_cooling = 0.1
        self.total_cost = 0.0

        # A (likelihood): vib in "g"
        self.mu_vibration = torch.tensor([0.025, 0.120, 0.120])
        self.sigma_vibration = torch.tensor([0.01, 0.03, 0.03])

        self.mu_temperature = torch.tensor([55.0, 50.0, 75.0])
        self.sigma_temperature = torch.tensor([2.0, 3.0, 4.0])

        # Preferenze (C) HOMEOSTATICHE: vib + temp 
        self.pref_mu_vib = 0.025
        self.pref_sigma_vib = 0.08

        self.pref_mu_temp = 55.0
        self.pref_sigma_temp = 10.0

        # pesi
        self.w_epistemic = 1.0
        self.w_pragmatic = 1.0

        # B (transitions): column-stochastic
        self.B: Dict[int, torch.Tensor] = {}

        self.B[0] = torch.tensor([
            [0.990, 0.05, 0.00],
            [0.005, 0.95, 0.00],
            [0.005, 0.00, 1.00],
        ])

        self.B[1] = self.B[0].clone()

        self.B[3] = torch.tensor([
            [0.98, 0.00, 0.95],
            [0.01, 1.00, 0.00],
            [0.01, 0.00, 0.05],
        ])

        self.eps = 1e-8
        self._sqrt_2pi = math.sqrt(2.0 * math.pi)

        self.likelihood_beta = 0.75

        # debug
        self.last_temperature: Optional[float] = None
        self.last_info: Dict = {}

        # per VFE
        self.last_prior = self.belief.clone()
        self.last_likelihood = torch.ones_like(self.belief)

    # ---------- utilities ----------
    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * self._sqrt_2pi)

    def belief_entropy(self, q=None):
        if q is None:
            q = self.belief
        q = torch.clamp(q, min=1e-12)
        return -torch.sum(q * torch.log(q))

    # ---------- inference ----------
    def compute_likelihood(self, observation):
        vibration = observation[0]
        p = self.gaussian(vibration, self.mu_vibration, self.sigma_vibration)

        # usa temperatura SOLO se presente (sensore attivo)
        if self.sensor.temperature_sensor_activation and len(observation) > 1:
            temperature = observation[1]
            self.last_temperature = float(temperature)

            p_temp = self.gaussian(temperature, self.mu_temperature, self.sigma_temperature)
            p = p * p_temp

            # spegni subito dopo la lettura
            self.sensor.temperature_sensor_activation = False

        return torch.clamp(p, min=1e-12)

    def update_belief(self, observation):
        prior = self.B[self.last_action] @ self.belief
        prior = torch.clamp(prior, min=self.eps)
        prior = prior / prior.sum()

        likelihood = self.compute_likelihood(observation)
        likelihood = torch.pow(likelihood, self.likelihood_beta)
        likelihood = torch.clamp(likelihood, min=1e-12)

        posterior = likelihood * prior
        posterior = posterior + self.eps
        posterior = posterior / posterior.sum()
        self.belief = posterior

        self.last_prior = prior.detach()
        self.last_likelihood = likelihood.detach()

    def compute_vfe(self, q=None, prior=None, likelihood=None):
        if q is None:
            q = self.belief
        if prior is None:
            prior = self.last_prior
        if likelihood is None:
            likelihood = self.last_likelihood

        q = torch.clamp(q, min=1e-12)
        prior = torch.clamp(prior, min=1e-12)
        likelihood = torch.clamp(likelihood, min=1e-12)

        log_q = torch.log(q)
        log_prior = torch.log(prior)
        log_like = torch.log(likelihood)

        vfe = torch.sum(q * (log_q - log_like - log_prior))
        complexity = torch.sum(q * (log_q - log_prior))
        accuracy = torch.sum(q * log_like)

        info = {
            "vfe": float(vfe.item()),
            "vfe_complexity": float(complexity.item()),
            "vfe_accuracy": float(accuracy.item()),
        }
        return float(vfe.item()), info

    # ---------- preferences (C) helpers ----------
    def expected_log_pref_gaussian(self, mu_s, sigma_s, pref_mu, pref_sigma):
        """
        E_{x~N(mu_s,sigma_s)}[ log N(x; pref_mu, pref_sigma) ]
        """
        pref_sigma_t = torch.tensor(pref_sigma, dtype=torch.float32)
        pref_mu_t = torch.tensor(pref_mu, dtype=torch.float32)

        mu_s = mu_s.to(torch.float32)
        sigma_s = sigma_s.to(torch.float32)

        const = -0.5 * torch.log(2.0 * torch.tensor(math.pi) * (pref_sigma_t ** 2))
        quad = -0.5 * (sigma_s**2 + (mu_s - pref_mu_t)**2) / (pref_sigma_t ** 2)
        return const + quad

    # ---------- battery costs (solo per fisica, NON per EFE) ----------
    def action_energy_cost(self, action: int) -> float:
        if action == 1:
            return float(self.cost_activate_temp_sensor)
        if action == 3:
            return float(self.cost_cooling)
        return 0.0

    # ---------- pragmatic gain (HOMEOSTATIC) ----------
    def compute_pragmatic_gain_homeostatic(self, action: int, horizon: int = 1) -> float:
        """
        PG(a) = sum_{tau=1..horizon} E[ log Q(o_{t+tau} | C) ]
        where Q(o|C) è la preferenza (homeostatic).
        """
        q = self.belief.clone()

        # vib sempre
        per_state_score = self.expected_log_pref_gaussian(
            self.mu_vibration, self.sigma_vibration,
            self.pref_mu_vib, self.pref_sigma_vib
        )

        # temp solo se osservabile sotto quell'azione
        
        per_state_score = per_state_score + self.expected_log_pref_gaussian(
            self.mu_temperature, self.sigma_temperature,
            self.pref_mu_temp, self.pref_sigma_temp
            )

        total_pg = 0.0
        for _ in range(horizon):
            q = self.B[action] @ q
            q = torch.clamp(q, min=self.eps)
            q = q / q.sum()
            total_pg = total_pg + torch.sum(q * per_state_score)

        return float(total_pg.item())

    # ---------- epistemic gain ----------
    def sample_observation_given_state(self, s_idx: int, include_temp: bool) -> torch.Tensor:
        vib = torch.normal(self.mu_vibration[s_idx], self.sigma_vibration[s_idx])
        vib = torch.clamp(vib, min=0.0)
        if include_temp:
            temp = torch.normal(self.mu_temperature[s_idx], self.sigma_temperature[s_idx])
            return torch.stack([vib, temp])
        return torch.stack([vib])

    def likelihood_of_observation(self, obs: torch.Tensor, include_temp: bool) -> torch.Tensor:
        vib = obs[0]
        like = self.gaussian(vib, self.mu_vibration, self.sigma_vibration)
        if include_temp:
            temp = obs[1]
            like = like * self.gaussian(temp, self.mu_temperature, self.sigma_temperature)
        return torch.clamp(like, min=1e-12)

    def compute_epistemic_gain(self, action: int, n_samples: int = 32) -> float:
        """
        IG(a) ≈ E_{o~Q(o|a)}[ KL( Q(s|o,a) || Q(s|a) ) ]
        include_temp SOLO se action==1 (perché attivi il sensore).
        """
        include_temp = (action == 1)

        prior = self.B[action] @ self.belief
        prior = torch.clamp(prior, min=self.eps)
        prior = prior / prior.sum()

        # se prior collassa, IG ~ 0
        if float(self.belief_entropy(prior).item()) < 1e-6:
            return 0.0

        ig_sum = 0.0
        prior_np = prior.detach().cpu()

        for _ in range(n_samples):
            s_idx = int(torch.multinomial(prior_np, 1).item())
            obs = self.sample_observation_given_state(s_idx, include_temp=include_temp)

            like = self.likelihood_of_observation(obs, include_temp=include_temp)

            post = like * prior
            post = torch.clamp(post, min=self.eps)
            post = post / post.sum()

            kl = torch.sum(post * (torch.log(post) - torch.log(prior)))
            ig_sum += float(kl.item())

        return ig_sum / float(n_samples)

    # ---------- EFE ----------
    def calculate_EFE(self, action: int, horizon: int = 1, n_ig_samples: int = 32) -> float:
      
        pg = self.compute_pragmatic_gain_homeostatic(action, horizon=horizon)
        ig = self.compute_epistemic_gain(action, n_samples=n_ig_samples)
        efe = -(self.w_epistemic * ig + self.w_pragmatic * pg)
        return float(efe)

    def select_action_via_efe(self, horizon: int = 1, n_ig_samples: int = 32) -> int:
        candidate_actions = [0, 1, 3]

        # 1) calcola EFE per ogni azione
        efe_vals = []
        for a in candidate_actions:
            if float(self.battery.level) < self.action_energy_cost(a):
                efe_vals.append(1e9)  # impossibile
            else:
                efe_vals.append(self.calculate_EFE(a, horizon=horizon, n_ig_samples=n_ig_samples))

        efe_t = torch.tensor(efe_vals, dtype=torch.float32)

        # 2) prior sulle azioni (log P(a)) -> penalizza 1 e 3
        # valori più negativi = meno probabile
        log_prior = torch.tensor([0.0, -2.0, -8.0], dtype=torch.float32)  # [a0,a1,a3]

        # 3) precisione/temperatura della softmax
        gamma = 4.0  # più alto => più “greedy” verso EFE minimo

        # 4) posterior Q(a) ∝ exp(-gamma*EFE + log_prior)
        logits = (-gamma * efe_t) + log_prior
        probs = torch.softmax(logits, dim=0)

        # 5) sample (stocastica) oppure MAP (deterministica)
        idx = torch.multinomial(probs, 1).item()      # stocastica
        # idx = int(torch.argmax(probs).item())       # deterministica MAP

        action = candidate_actions[idx]

        # debug
        print("EFE:", {a: float(efe_vals[i]) for i, a in enumerate(candidate_actions)})
        print("probs:", {a: float(probs[i]) for i, a in enumerate(candidate_actions)})
        print("chosen:", action)

        # monitoring
        self.last_info.update({
            "softmax/gamma": float(gamma),
            "softmax/p_a0": float(probs[0]),
            "softmax/p_a1": float(probs[1]),
            "softmax/p_a3": float(probs[2]),
            "softmax/chosen": int(action),
        })

        self.last_action = action
        return action


    # ---------- loop step ----------
    def process_data(self, data):
        self.last_info = {}

        # costo lettura vib (dinamica batteria reale)
        self.total_cost += self.cost_read_vibration
        self.battery.consume(self.cost_read_vibration)

        # update belief con osservazione reale (solo vib/temp)
        self.update_belief(data)

        # VFE
        _, vfe_info = self.compute_vfe()

        print(
            f"VFE: {vfe_info['vfe']:.4f} | "
            f"complexity: {vfe_info['vfe_complexity']:.4f} | "
            f"accuracy: {vfe_info['vfe_accuracy']:.4f}"
        )

        H = float(self.belief_entropy().item())
        print("Belief q(s):")
        for i, s in enumerate(self.states):
            print(f"  P({s}) = {self.belief[i]:.3f}")
        print(f"Entropy H(q): {H:.3f}")

        # ACTION SELECTION (EFE = -(IG + PG), PG coerente)
        action = self.select_action_via_efe(horizon=1, n_ig_samples=32)

        # --- vincolo fisico DOPO la scelta (così la batteria non entra nell'EFE) ---
        needed = self.action_energy_cost(action)
        if self.battery.level < needed:
            print(f"PHYSICS OVERRIDE: not enough battery for action={action} (need {needed:.2f}, have {self.battery.level:.2f}) -> action=0")
            self.last_info["physics/override"] = 1
            self.last_info["physics/override_from"] = int(action)
            action = 0
        else:
            self.last_info["physics/override"] = 0
            self.last_info["physics/override_from"] = -1

        # memorizza per prior prossimo step (coerente col mondo)
        self.last_action = action
        return action


# =========================
# ACTUATOR
# =========================
class actuator:
    def __init__(self, agent_ref: agent, sensor_ref: sensor):
        self.agent = agent_ref
        self.sensor = sensor_ref

    def take_action(self, action: int):
        if action == 1:
            if self.agent.battery.level >= self.agent.cost_activate_temp_sensor:
                print("Action: ACTIVATING temperature sensor.")
                self.sensor.temperature_sensor_activation = True
                self.agent.total_cost += self.agent.cost_activate_temp_sensor
                self.agent.battery.consume(self.agent.cost_activate_temp_sensor)
            else:
                print("Not enough battery to activate temperature sensor.")

        elif action == 3:
            if self.agent.battery.level >= self.agent.cost_cooling:
                print("Action: COOLING SYSTEM ACTIVATED!")
                self.agent.total_cost += self.agent.cost_cooling
                self.agent.battery.consume(self.agent.cost_cooling)

                self.sensor.persistent_anomaly_mean = torch.tensor([0.025, 55.0])
                self.sensor.persistent_duration = False
                self.sensor.anomaly_type = None

                print("Cooling complete → system back to NORMAL state.")
            else:
                print("Not enough battery to activate cooling system.")
        else:
            print("Action: nothing")


# =========================
# DRONE
# =========================
class drone:
    def __init__(self):
        self.battery = Battery(capacity=100.0)
        self.sensor = sensor()
        self.agent = agent(self.battery, self.sensor)
        self.actuator = actuator(self.agent, self.sensor)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    drone_instance = drone()

    countInteractions = 0
    global_step = 0

    while True:
        if keyboard.is_pressed("enter"):
            wandb.finish()
            print("Interrotto dall'utente.")
            break

        if countInteractions >= 100:
            if drone_instance.sensor.anomaly_type is None and not drone_instance.sensor.persistent_duration:
                print("INJECTING ANOMALY...")
                anomaly_type = ran.choice(["perturbation", "overheating"])
                drone_instance.sensor.anomalyInjection(anomaly_type)
            countInteractions = 0

        data = drone_instance.sensor.get_data()
        anomaly_str = drone_instance.sensor.anomaly_type
        print(f"Outputs {data[0].item()} anomaly={anomaly_str}")

        action = drone_instance.agent.process_data(data)
        drone_instance.actuator.take_action(action)

        payload = {
            "vibration_sensor_g": float(data[0]),
            "anomaly_type": anomaly_str if anomaly_str is not None else "none",

            "battery_level": float(drone_instance.battery.level),
            "battery_percentage": float(drone_instance.battery.percentage()),
            "total_cost": float(drone_instance.agent.total_cost),

            "action": int(action),

            "belief/normal": float(drone_instance.agent.belief[0].item()),
            "belief/perturbation": float(drone_instance.agent.belief[1].item()),
            "belief/overheating": float(drone_instance.agent.belief[2].item()),
            "belief_entropy": float(drone_instance.agent.belief_entropy().item()),

            "debug/perturbation_duration": int(drone_instance.sensor.perturbation_duration),
            "debug/persistent_duration": int(drone_instance.sensor.persistent_duration),
            "debug/last_temperature": float(drone_instance.agent.last_temperature)
            if drone_instance.agent.last_temperature is not None else -1.0,
        }

        if len(data) > 1:
            payload["temperature_sensor"] = float(data[1])

        payload.update({k: v for k, v in drone_instance.agent.last_info.items() if v is not None})

        wandb.log(payload, step=global_step)

        print(f"Battery: {drone_instance.battery.percentage():.2f}%")
        print(f"Total energy cost so far: {drone_instance.agent.total_cost}")
        print("-----")

        countInteractions += 1
        global_step += 1
        
