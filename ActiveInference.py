import math
import time
import keyboard
import wandb
import torch
import random as ran
from typing import Dict, Optional

wandb.login(key="50135bcebab208e2a5caa7a8cb9c1037e2abcd2c")
wandb.init(project="drone_telemetry")



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
        self.instant_anomaly_mean = torch.tensor([0.3, 50.0])
        self.instant_anomaly_std = torch.tensor([0.01, 2.0])

        # overheating (persistent)
        self.persistent_anomaly_mean = torch.tensor([0.2, 75.0])
        self.persistent_anomaly_std = torch.tensor([0.01, 3.0])
        self.persistent_duration = False

    def anomalyInjection(self, anomaly_type):
        if anomaly_type == "perturbation":
            self.anomaly_type = anomaly_type
            self.perturbation_duration = ran.randint(10, 20)
            tmp = ran.uniform(0.15, 0.3)
            self.instant_anomaly_mean = torch.tensor([tmp, 50.0])

        elif anomaly_type == "overheating":
            self.anomaly_type = anomaly_type
            self.persistent_duration = True
            self.persistent_anomaly_mean = torch.tensor([0.2, 75.0])

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
                # ultimo step: torna normale
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
# BATTERY
# =========================
class Battery:
    def __init__(self, capacity=100.0):
        self.capacity = capacity
        self.level = capacity

    def consume(self, amount):
        self.level -= amount
        self.level = max(self.level, 0.0)

    def is_empty(self):
        return self.level <= 0.0

    def percentage(self):
        return (self.level / self.capacity) * 100


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
        self.last_action = 0  # per prior predetto

        # costi
        self.cost_read_vibration = 0.2
        self.cost_activate_temp_sensor = 1.0
        self.cost_cooling = 10.0
        self.total_cost = 0.0

        # A (likelihood): mu vib in m/s^2
        self.mu_vibration = torch.tensor([
            0.025 * 9.81,   # normal
            0.25  * 9.81,   # perturbation
            0.22  * 9.81    # overheating (overlap)
        ])
        self.sigma_vibration = torch.tensor([0.06, 0.22, 0.22])


        self.mu_temperature = torch.tensor([55.0, 50.0, 75.0])
        self.sigma_temperature = torch.tensor([2.0, 3.0, 4.0])

        # C (preferences) per STATO
        self.C = torch.tensor([0.0, 0.0, -20.0])

        # B (transitions): column-stochastic
        self.B: Dict[int, torch.Tensor] = {}

        self.B[0] = torch.tensor([
            [0.90, 0.20, 0.05],
            [0.08, 0.60, 0.15],
            [0.02, 0.20, 0.80],
        ])

        # ✅ action 1 NON cambia lo stato: stesso B di "nothing"
        self.B[1] = self.B[0].clone()

        # cooling cambia le transizioni (porta verso normal)
        self.B[3] = torch.tensor([
            [0.95, 0.90, 0.85],
            [0.04, 0.08, 0.10],
            [0.01, 0.02, 0.05],
        ])

        self.eps = 1e-8
        self._sqrt_2pi = math.sqrt(2.0 * math.pi)

        # tempering likelihood (anti-collasso)
        self.likelihood_beta = 0.75

        # Monte Carlo IG
        self.temp_query_min_mass = 0.20
        self.mc_samples = 30

        # debug
        self.last_temperature: Optional[float] = None
        self.last_info: Dict = {}

    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * self._sqrt_2pi)

    def compute_likelihood(self, observation):
        vibration = observation[0] * 9.81
        p = self.gaussian(vibration, self.mu_vibration, self.sigma_vibration)

        # usa temperatura SOLO se presente
        if self.sensor.temperature_sensor_activation and len(observation) > 1:
            temperature = observation[1]
            self.last_temperature = float(temperature)

            p_temp = self.gaussian(temperature, self.mu_temperature, self.sigma_temperature)
            p = p * p_temp

            # spegni subito dopo la lettura
            self.sensor.temperature_sensor_activation = False

        return torch.clamp(p, min=1e-12)

    def update_belief(self, observation):
        # prior predetto con B e last_action (filtering POMDP)
        prior = self.B[self.last_action] @ self.belief
        prior = torch.clamp(prior, min=self.eps)
        prior = prior / prior.sum()

        likelihood = self.compute_likelihood(observation)
        likelihood = torch.pow(likelihood, self.likelihood_beta)

        self.belief = likelihood * prior
        self.belief = self.belief + self.eps
        self.belief = self.belief / self.belief.sum()

    def belief_entropy(self, q=None):
        if q is None:
            q = self.belief
        q = torch.clamp(q, min=1e-12)
        return -torch.sum(q * torch.log(q))

    def _expected_information_gain_modality(self, q_prior, modality: str, n_samples: int):
        q_prior = q_prior / q_prior.sum()

        # se già certo, IG ~ 0
        if torch.max(q_prior).item() > 0.98:
            return 0.0

        kl_sum = 0.0
        for _ in range(n_samples):
            s_idx = torch.multinomial(q_prior, 1, replacement=True).item()

            if modality == "vibration":
                o = torch.normal(self.mu_vibration[s_idx], self.sigma_vibration[s_idx])
                like = self.gaussian(o, self.mu_vibration, self.sigma_vibration)
            else:
                o = torch.normal(self.mu_temperature[s_idx], self.sigma_temperature[s_idx])
                like = self.gaussian(o, self.mu_temperature, self.sigma_temperature)

            q_post = q_prior * torch.clamp(like, min=1e-12)
            q_post = q_post / q_post.sum()

            kl = torch.sum(q_post * (torch.log(q_post + self.eps) - torch.log(q_prior + self.eps)))
            kl_sum += float(kl.item())

        return kl_sum / float(n_samples)

    def compute_efe(self, action: int):
        q_next = self.B[action] @ self.belief
        q_next = torch.clamp(q_next, min=self.eps)
        q_next = q_next / q_next.sum()

        ig_vib = self._expected_information_gain_modality(q_next, "vibration", self.mc_samples)

        ig_temp = 0.0
        ambig_mass = float((q_next[1] + q_next[2]).item())
        if action == 1:
            if ambig_mass >= self.temp_query_min_mass and torch.max(q_next).item() < 0.98:
                ig_temp = self._expected_information_gain_modality(q_next, "temperature", self.mc_samples)

        epistemic_gain = ig_vib + ig_temp

        # pragmatic gain = E_q[C]
        pragmatic_gain = float(torch.sum(q_next * self.C).item())

        cost = {0: 0.0, 1: self.cost_activate_temp_sensor, 3: self.cost_cooling}[action]

        # minimizza G
        G = - epistemic_gain - pragmatic_gain + cost


        info = {
            "efe": float(G),
            "epistemic_gain": float(epistemic_gain),
            "ig_vibration": float(ig_vib),
            "ig_temperature": float(ig_temp),
            "pragmatic_gain": float(pragmatic_gain),
            "cost": float(cost),
        }
        return float(G), info

    def select_action_via_efe(self):
        
        if self.battery.is_empty():
            return 0

        candidates = [0, 1, 3]
        feasible = []

        for a in candidates:
            if a == 1:
                if self.sensor.temperature_sensor_activation:
                    continue
                if self.battery.level < self.cost_activate_temp_sensor:
                    continue

            if a == 3 and self.battery.level < self.cost_cooling:
                continue

            feasible.append(a)

        if not feasible:
            return 0

        Gs = {}
        infos = {}
        for a in feasible:
            g, info = self.compute_efe(a)
            Gs[a] = g
            infos[a] = info

        best_action = min(Gs, key=Gs.get)

        print("EFE values:", {a: float(Gs[a]) for a in Gs})

        self.last_info = {
            "EFE/nothing": Gs.get(0, None),
            "EFE/activate_temp": Gs.get(1, None),
            "EFE/cooling": Gs.get(3, None),
            "EFE/best_action": int(best_action),
            "EFE/best_value": float(Gs[best_action]),
            "best/epistemic_gain": infos[best_action]["epistemic_gain"],
            "best/ig_vibration": infos[best_action]["ig_vibration"],
            "best/ig_temperature": infos[best_action]["ig_temperature"],
            "best/pragmatic_gain": infos[best_action]["pragmatic_gain"],
            "best/cost": infos[best_action]["cost"],
        }

        return best_action

    def process_data(self, data):
        if self.battery.is_empty():
            print("Battery empty → agent inactive")
            return 0

        self.total_cost += self.cost_read_vibration
        self.battery.consume(self.cost_read_vibration)

        self.update_belief(data)

        H = float(self.belief_entropy().item())
        print("Belief q(s):")
        for i, s in enumerate(self.states):
            print(f"  P({s}) = {self.belief[i]:.3f}")
        print(f"Entropy H(q): {H:.3f}")

        action = self.select_action_via_efe()

        # salva azione per il prior al prossimo step
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
            "vibration_sensor": float(data[0]) * 9.81,
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
            "debug/last_temperature": float(drone_instance.agent.last_temperature) if drone_instance.agent.last_temperature is not None else -1.0,
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
        time.sleep(0.05)
