
## Indice

| Introduzione ...................................................................................................................... 1                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Capitolo 1: Impostazione del lavoro e quadro sperimentale ...................................................... 2                                                                                                            |
| 1.1 Architettura del sistema e ciclo di interazione ................................................................ 2                                                                                                        |
| 1.2 Definizione operativa del problema .............................................................................. 3                                                                                                       |
| 1.3 Variabili del problema: stati latenti, osservazioni, azioni .................................................. 3                                                                                                          |
| 1.4 Caso di studio: telemetria e anomalie (perturbation vs overheating) .................................. 4                                                                                                                  |
| 1.5 Metodi a confronto ( overview ): ................................................................................... 5                                                                                                    |
| 1.5.1 Baseline deterministica ........................................................................................ 5                                                                                                      |
| 1.5.2 Inferenza esplicita + EFE (matrici stocastiche) ........................................................ 6                                                                                                              |
| 1.5.3 Inferenza ammortizzata con PosteriorNet + EFE ...................................................... 6                                                                                                                  |
| 1.6 Domande di ricerca e ipotesi ....................................................................................... 6                                                                                                    |
| 1.7 Metriche e logging (W&B) ......................................................................................... 7                                                                                                      |
| Capitolo 2: Formalizzazione: modello generativo, VFE ed EFE ................................................ 8                                                                                                                |
| 2.1 Notazione e formulazione generale .............................................................................. 8                                                                                                        |
| 2.2 Modello generativo dell'agente ................................................................................... 9                                                                                                      |
| 2.2.1 Likelihood 𝑝(𝑜𝑡 ∣ 𝑠𝑡) ........................................................................................... 9                                                                                                     |
| 2.2.2 Transizioni e matrici 𝐵𝑎 .....................................................................................                                                                                                          |
| 10 2.2.3 Preferenze (omeostatica) e funzione di utilità sulle osservazioni ............................... 11                                                                                                                 |
| 11                                                                                                                                                                                                                            |
| 2.3 Variational Free Energy (VFE): ruolo nell'inferenza e come viene calcolata .....................                                                                                                                          |
| 2.3.2 Entropia del belief e relazione con incertezza ........................................................ 12                                                                                                              |
| 2.4 Expected Free Energy (EFE) per selezione dell'azione .................................................. 13                                                                                                                |
| 2.4.1 Componente epistemica (Information Gain) .......................................................... 13                                                                                                                  |
| 2.4.2 Componente pragmatica (raggiungimento preferenze) ............................................ 14 2.5 Inferenza ammortizzata: PosteriorNet ......................................................................... 15 |
| Capitolo 3: Metodologia e implementazione dei metodi ......................................................... 17                                                                                                             |
| 3.1 Metodo A-Inferenza esplicita del posteriore (Bayes update) ....................................... 17                                                                                                                     |
| 3.1.1 Calcolo likelihood e gestione osservazioni parziali ................................................. 17                                                                                                                |
| 3.1.2 Aggiornamento del belief 𝑞(𝑠𝑡) ........................................................................... 18                                                                                                           |
| 3.1.3 Logging: VFE, entropia, belief per stato ............................................................... 19                                                                                                             |

| 3.2 Metodo B -PosteriorNet (amortized inference) .........................................................                         |   19 |
|------------------------------------------------------------------------------------------------------------------------------------|------|
| 3.2.1 Feature design ..................................................................................................            |   19 |
| 3.2.2 Replay buffer, loss cross-entropy, ottimizzazione ...................................................                        |   20 |
| 3.2.3 Uso del belief neurale nella pipeline EFE ..............................................................                     |   20 |
| 3.3 Selezione dell'azione via EFE ...................................................................................              |   21 |
| 3.3.1 Calcolo IG .......................................................................................................           |   22 |
| 3.3.2 Calcolo PG .......................................................................................................           |   23 |
| 3.3.3 Softmax/argmax ................................................................................................              |   23 |
| 3.4 Vincoli fisici separati dal criterio decisionale ...............................................................               |   24 |
| Capitolo 4: Metodi sperimentali e risultati ...........................................................................            |   25 |
| 4.1 Setup sperimentale ..................................................................................................          |   25 |
| 4.2 Analisi congiunta: belief, VFE e decomposizione accuracy/complexity ...........................                                |   26 |
| 4.3 Analisi EFE e selezione dell'azione ............................................................................               |   28 |
| 4.4 PosteriorNet e confronto qualitativo con inferenza esplicita ...........................................                       |   31 |
| Conclusioni .....................................................................................................................  |   33 |
| Bibliografia ..................................................................................................................... |   34 |

## Introduzione

La tesi mette a confronto due approcci per la progettazione di sistemi di decisione in scenari IoT e in particolare su un caso di studio centrato sugli Unmanned Aerial Vehicles (UAV) con approccio simbolico e approccio subsimbolico. Per approccio simbolico si intende un metodo deterministico che è basato su rappresentazioni esplicite, come per esempio soglie fissate a mano. Questo approccio solitamente beneficia di trasparenza, controllabilità e rapidità di esecuzione, ma può essere vulnerabile quando i sensori sono rumorosi e l'ambiente è incerto. L'approccio subsimbolico, invece, comprende le reti neurali, in grado di apprendere dai dati modelli complessi. Questo secondo approccio tende a essere più flessibile e robusto in ambienti rumorosi, ma può risultare meno interpretabile e più difficile da vincolare con requisiti di sicurezza e affidabilità.

In particolare, è stato scelto un UA V come caso di studio poiché sono sistemi IoT caratterizzati da sensori che producono telemetrie e da un'elevata esposizione a incertezza e variabili d'ambiente. Proprio queste caratteristiche li rendono adatti a mettere alla prova approcci di decisione e inferenza e, inoltre, possono essere interessanti per applicazioni future.

L'obiettivo di questo lavoro è quindi valutare sperimentalmente i punti di forza e le debolezze dei due approcci in un contesto realistico e controllato, evidenziando anche la ragione e il potenziale di una prospettiva Hybrid AI, intesa come fusione tra una parte strutturata e controllabile (ragionamento esplicito, regole, vincoli e logiche di decisione deterministiche) e una parte apprendente (apprendimento statistico, inferenza approssimata, generalizzazione).

La tesi è strutturata come segue: dopo aver presentato la struttura e i concetti teorici di base, viene descritto il caso di studio UA V e la modellazione dell'ambiente, quindi vengono introdotti gli approcci considerati: baseline simbolica deterministica, approccio subsimbolico e configurazione ibrida e le metriche di valutazione. Infine, vengono presentati e discussi i risultati sperimentali, mettendo in luce i trade-off osservati e le riflessioni per la progettazione di sistemi autonomi   affidabili.

## Capitolo 1: Impostazione del lavoro e quadro sperimentale

## 1.1 Architettura del sistema e ciclo di interazione

Il caso di studio è modellato come un sistema composto da tre moduli concettualmente distinti: mondo (processo generativo), agente (inferenza + decisione), attuatore (applicazione delle azioni). Questa separazione è necessaria per distinguere ciò che è generato dal mondo da ciò che viene stimato e controllato dall'agente. [1]

Il mondo è composto da un modulo di sensing che genera dati simulando la telemetria e che include dinamiche anomale che simulano l'inserimento di rumore all'interno del mondo ogni N numero di interazioni. Inoltre, il mondo include una risorsa fisica che è la batteria che si riduce nel tempo e vincola l'esecuzione di certe azioni.

L'agente riceve i dati sensoriali e mantiene una rappresentazione interna probabilistica dello stato. Sulla base delle credenze e di un criterio decisionale, seleziona l'azione successiva e manda un impulso (un intero in questo caso) all'attuatore su quale azione eseguire.

Il modulo di attuazione in base all'impulso ricevuto applica l'azione al mondo, modificando lo stato del sistema o la disponibilità di informazione per i passi successivi (ad esempio attivando il sensore della temperatura o un meccanismo di raffreddamento).

Il ciclo temporale dell'interazione può essere descritto come segue:

1. Il mondo produce un'osservazione 𝑜𝑡 tramite i sensori (modulo sensoriale).
2. L'agente aggiorna le proprie credenze 𝑞(𝑠𝑡 ) a partire da 𝑜𝑡 e dalle dinamiche assunte.
3. L'agente seleziona un'azione 𝑎𝑡 tra quelle disponibili.
4. L'attuatore applica 𝑎𝑡 al mondo, influenzando la generazione di 𝑜𝑡+1 e/o le transizioni di stato.
5. Successivamente si riprende dal punto 1.

Questa struttura rende esplicito il ruolo di ciascun componente e permette di interpretare correttamente sia le grandezze inferenziali (credenza, entropia, VFE) che i comportamenti decisionali (scelte d'azione guidate da EFE o regole deterministiche).

## 1.2 Definizione operativa del problema

Il lavoro è impostato come un confronto sperimentale tra tre strategie di selezione dell'azione in un sistema con osservazioni parziali e incertezza. Il confronto isola l'effetto del metodo di inferenza/decisione mantenendo invariati scenario, interfacce e vincoli.

In seguito, definiamo due concetti fondamentali per la tipologia di approccio adottata:

- Simbolico: decisione determinata da strutture esplicite e ispezionabili (regole/soglie) definite a priori.
- Subsimbolico: rappresentazione e/o inferenza apprese in forma distribuita (parametri), tipicamente tramite rete neurale.

Sulla base di tali definizioni, i metodi confrontati sono: baseline deterministica a soglie (simbolico), Active  Inference  con  inferenza  esplicita  sulle  credenze  e  selezione  azione  tramite Expected Free Energy (EFE), Active Inference con inferenza ammortizzata tramite rete neurale del posteriore e selezione tramite EFE (ibrido con componente subsimbolica). L'obiettivo è valutare trade-off tra robustezza, riduzione dell'incertezza e costi (energetici e computazionali), tramite metriche uniformi. [2]

## 1.3 Variabili del problema: stati latenti, osservazioni, azioni

All'interno del ciclo descritto, si introduce una formalizzazione a variabili discrete per rappresentare il regime operativo del sistema.[3] Lo stato latente al tempo 𝑡 è definito come:

st ∈ {normal, perturbation, overheating}.

Lo stato è detto 'latente' perché non è osservato direttamente dall'agente [4] ciò che l'agente riceve è un'osservazione 𝑜𝑡 generata dal mondo. L'osservazione include sempre una misura di vibrazione

𝑣𝑡 e può includere una misura di temperatura 𝑇𝑡 quando il sensore addizionale di temperatura è attivo. In particolare,

- se il sensore addizionale non è attivo allora si avrà un'osservazione del tipo: 𝑜𝑡 = [𝑣𝑡 ] ;
- mentre nel caso contrario l'osservazione sarebbe: 𝑜𝑡 = [𝑣𝑡 , 𝑇 𝑡 ] .

L'agente seleziona azioni discrete:

<!-- formula-not-decoded -->

dove 𝑎 = 0 rappresenta un'azione neutra (nessun intervento), 𝑎 = 1 l'attivazione del sensore di temperatura (azione epistemica), e 𝑎 = 3 l'attivazione del sistema di raffreddamento (azione pragmatica). L'effetto delle azioni è mediato dall'attuatore e si manifesta nel mondo influenzando la disponibilità dell'osservazione e/o le transizioni di stato.

Un aspetto centrale dello scenario è la presenza di un vincolo fisico di risorsa: la batteria del sistema si riduce nel tempo e limita l'eseguibilità di alcune azioni. Nel lavoro questo vincolo viene trattato esplicitamente come dinamica del mondo e applicato come vincolo post-decisione, distinguendo il criterio di decisione (metodo) dai limiti fisici.

L'obiettivo generale del confronto non è la progettazione di un sistema UAV completo, ma lo studio controllato di un problema di selezione dell'azione sotto incertezza, con un caso di studio che rende misurabili i trade-off tra: (i) robustezza alle anomalie e al rumore, (ii) riduzione dell'incertezza, (iii) costo energetico e costo computazionale.

## 1.4 Caso di studio: telemetria e anomalie (perturbation vs overheating)

L'obiettivo dell'ambiente sperimentale è introdurre due classi di comportamento anomalo con proprietà temporali differenti, così da rendere misurabile robustezza, gestione dell'incertezza e strategie d'azione. In particolare, lo scenario distingue un regime nominale stabile e due regimi anomali con dinamiche diverse, consentendo di analizzare sia stati transitori che condizioni persistenti.

In condizioni di assenza di anomalie, la telemetria è generata attorno a valori medi di riferimento con variabilità stocastica. In particolare, la vibrazione rimane prossima al regime nominale con rumore, e la temperatura oscilla attorno a un valore di riferimento. Queste distribuzioni definiscono il comportamento atteso del sistema in assenza di anomalie e costituiscono la baseline del processo generativo.

Successivamente, vengono definite le caratteristiche delle due anomalie.

La perturbation è un evento a durata finita che altera temporaneamente la distribuzione della telemetria (in particolare aumentando la vibrazione rispetto al nominale). Dopo un numero limitato di step l'anomalia termina e il processo rientra spontaneamente nel regime normale.

Questo caso permette di valutare la capacità dei metodi di:

- riconoscere deviazioni brevi senza attivare interventi eccessivi;
- discriminare tra fluttuazione e anomalia mediante acquisizione informativa quando disponibile.

L' overheating è modellato come un regime anomalo persistente, associato a un incremento della temperatura e a una dinamica che tende a mantenere l'anomalia nel tempo. A differenza della perturbation , il rientro alle condizioni nominali non è automatico: l'anomalia persiste finché non viene applicata un'azione correttiva (sistema di cooling ). Questo caso consente di valutare:

- la capacità dei metodi di riconoscere condizioni persistenti che richiedono intervento;
- la gestione del trade-off tra acquisizione di informazione e intervento correttivo.

## 1.5 Metodi a confronto ( overview ):

In questa sezione si riassumono i tre metodi implementati e confrontati, introdotti precedentemente, i dettagli teorici e implementativi saranno sviluppati nei capitoli successivi.

## 1.5.1 Baseline deterministica

Il  primo  modello implementato è una baseline rule-based che  seleziona  l'azione tramite regole e treshold definite a priori sulle osservazioni disponibili.[5] La baseline è utilizzata come termine di paragone perché ha costo computazionale minimo e comportamento facilmente interpretabile, ma non rappresenta esplicitamente l'incertezza sullo stato latente né implementa un criterio principled per il bilanciamento tra acquisizione informativa e intervento correttivo. Nel confronto sperimentale il suo ruolo è evidenziare quando e quanto l'adozione di un modello probabilistico e/o di una componente appresa porti benefici misurabili.

## 1.5.2 Inferenza esplicita + EFE (matrici stocastiche)

Nella prima implementazione l'agente mantiene una distribuzione di credenze 𝑞(𝑠𝑡 ) sullo stato latente che viene aggiornato tramite inferenza esplicita basata su un modello generativo. La dinamica è rappresentata da matrici di transizione stocastiche per ogni azione, mentre la relazione stato-osservazione è modellata tramite una verosimiglianza parametrizzata.

La selezione dell'azione avviene tramite la minimizzazione dell'Expected Free Energy, che viene calcolata grazie all'utilizzo del guadagno epistemico e del guadagno pragmatico. Questo metodo rappresenta la variante completamente model-based, in cui sia inferenza che decisione sono guidate da quantità esplicite. [6] [7]

## 1.5.3 Inferenza ammortizzata con PosteriorNet + EFE

Il secondo metodo mantiene invariato il criterio decisionale basato su EFE, ma sostituisce il criterio di aggiornamento delle credenze sugli stati con un modello neurale che stima direttamente la posterior 𝑞𝜃(𝑠𝑡 ∣ 𝑜 𝑡 , 𝑎 𝑡-1 ) . La rete riceve come input un'osservazione e l'azione consumata precedentemente, producendo una distribuzione sui possibili stati latenti. In questo  modo è possibile  confrontare il trade-off e la qualità dell'inferenza, osservando come una stima subsimbolica delle credenze influenzi la selezione d'azione guidata da EFE.

## 1.6 Domande di ricerca e ipotesi

Le domande di ricerca mirano a confrontare e stabilire se un approccio simbolico differisce con l'utilizzo di un approccio model-based o subsimbolico. In particolare, quello che ci chiediamo è:

- i. Quanto i tre approcci differiscono nella gestione dell'incertezza e nella stabilità del comportamento durante anomalie transitorie e persistenti?
- ii. L'approccio con EFE migliora la selezione delle azioni rispetto ad un approccio baseline con treshold?
- iii. La PosteriorNet consente di ridurre costo computazionale mantenendo prestazioni comparabili rispetto all'inferenza esplicita?

In base a queste tre domande formuliamo queste ipotesi:

- i. Per la prima domanda si suppone che la baseline a soglie sia meno robusta e più sensibile a rumore e ad osservazioni parziali;
- ii. Per la seconda invece si ipotizza che l'uso di EFE migliori il comportamento decisionale grazie alla combinazione di valore epistemico e pragmatico;
- iii. Infine, per la terza, l'inferenza ammortizzata riduca il costo per step mantenendo prestazioni comparabili entro un margine misurabile.

## 1.7 Metriche e logging (W&amp;B)

Per valutare le domande di ricerca vengono registrate metriche raggruppate in: metriche di inferenza, includendo la distribuzione di credenza q(st ) e l'entropia H(q) come misura di incertezza; metriche basate su Free Energy, monitorando la Variational Free Energy (VFE) e la sua decomposizione in accuracy e complexity ; metriche decisionali, includendo l'azione selezionata e la frequenza/propensione verso azioni informative e correttive e i valori di Expected Free Energy (EFE) associati alle azioni candidate; metriche di costo e vincoli, quali livello batteria, costo energetico cumulativo e occorrenze di override fisico. Tutti i seguenti dati vengono raccolte su W&amp;B. [8]

## Capitolo 2: Formalizzazione: modello generativo, VFE ed EFE

## 2.1 Notazione e formulazione generale

In questa sezione introduciamo una notazione generale, già utilizzata in precedenza nel capitolo 1 per descrivere osservazioni, stati latenti e azioni, per la definizione di problemi di inferenza e selezione dell'azione. Nel seguito si adotterà una formulazione astratta, che verrà poi istanziata sul caso sperimentale.

Si consideri un processo a tempo discreto t = 1, … , T caratterizzato da:

- uno stato latente st ∈ S , non direttamente osservabile;
- un'osservazione ot ∈ O , generata dallo stato e disponibile all'agente;
- un'azione at ∈  A , selezionata dall'agente e applicata al sistema.

L'agente mantiene una distribuzione di credenza sullo stato latente, indicata con:

<!-- formula-not-decoded -->

che rappresenta un'approssimazione del posteriore e costituisce la variabile centrale sia per l'inferenza (minimizzazione della Free Energy variazionale) sia per la decisione (valutazione dell'Expected Free Energy).

Nel seguito si assume una struttura Markoviana standard:

<!-- formula-not-decoded -->

ossia lo stato futuro dipende solo dallo stato corrente e dall'azione, e l'osservazione corrente dipende solo dallo stato corrente. Questa assunzione consente di definire un modello generativo fattorizzato e di derivare in modo diretto le quantità di interesse (VFE ed EFE) introdotte nelle sezioni successive.

## 2.2 Modello generativo dell'agente

Il modello generativo interno all'agente specifica come lo stato latente evolva nel tempo e come generi le osservazioni. [13] In forma generale, per una sequenza di stati 𝑠0:𝑇 , osservazioni 𝑜0:𝑇 e azioni 𝑎0:𝑇-1 , si assume una fattorizzazione Markoviana:

<!-- formula-not-decoded -->

Questa forma separa in modo esplicito:

1. un modello di osservazione p(ot ∣ s t ) ,
2. un modello di transizione p(st+1 ∣ s t , a t ) ,
3. eventuali preferenze sulle osservazioni, che non descrivono 'come il mondo genera i dati', ma 'quali osservazioni sono desiderabili' per l'agente.

È inoltre importante distinguere il concetto di modello generativo dal concetto di processo generativo:

Quando si parla di processo generativo, si intende il mondo, cioè colui che genera realmente i dati che vengono osservati e sul quale vengono eseguite azioni. Mentre il modello generativo si trova all'interno dell'agente ed è la rappresentazione/ipotesi dell'agente su quel processo.

## 2.2.1 Likelihood 𝑝(𝑜𝑡 ∣ 𝑠 𝑡 )

Si assume che, condizionatamente allo stato, vibrazione e temperatura siano (approssimativamente) indipendenti:

<!-- formula-not-decoded -->

Il termine 𝑝(𝑇𝑡 ∣ 𝑠 𝑡 ) 𝑚𝑡 implementa l'osservazione parziale: se 𝑚𝑡 = 0 , il fattore sulla temperatura vale 1 e quindi non contribuisce al calcolo della likelihood .

Inoltre, le componenti sono modellate come gaussiane quindi si ha che:

<!-- formula-not-decoded -->

## 2.2.2 Transizioni e matrici 𝐵𝑎

Poiché 𝑠𝑡 è  discreto,  il  modello  di  transizione  viene  rappresentato  tramite  matrici  dipendenti dall'azione:

<!-- formula-not-decoded -->

Queste matrici sono a colonne-stocastiche, cosicché, data la credenza corrente 𝑞𝑡 , il prior predittivo sullo stato successivo data l'azione 𝑎 è:

<!-- formula-not-decoded -->

Questa equazione formalizza la parte dinamica del modello generativo in modo che l'azione selezionata influenza la distribuzione su stati futuri.

È importante fare una distinzione tra le azioni che influenzano la dinamica e le funzioni che influenzano l'informazione.

Nel caso di studio abbiamo due azioni, sensing della temperatura e attivazione del cooling system , che si distinguono in:

- Azione epistemica: aumentano la qualità/dimensione dell'osservazione (ad es.  modificano 𝑚𝑡 ), senza cambiare direttamente lo stato latente;
- Azione pragmatica: modificano la probabilità di transizione tra stati (o forzano un rientro verso il nominale).

## 2.2.3 Preferenze (omeostatica) e funzione di utilità sulle osservazioni

Oltre a descrivere 'come il mondo genera i dati', l'agente deve codificare 'quali esiti desidera'. In Active  Inference  questo  è  espresso  tramite  preferenze  sulle  osservazioni,  spesso  indicate  con 𝐶 o 𝑝𝐶(𝑜) . Operativamente, si introduce una distribuzione preferita sulle osservazioni 𝑝𝐶(𝑜𝑡) tale che la densità risulti maggiore se 𝑜𝑡 si trova vicino alle osservazioni desiderate.

Nel caso omeostatico, l'agente preferisce osservazioni vicine a valori nominali di vibrazione e temperatura. Nel mio caso sperimentale applico una scelta naturale che è quella definire preferenze gaussiane:

<!-- formula-not-decoded -->

Analogamente alla likelihood, quando la temperatura non è osservata si può ignorarne il contributo tramite 𝑚𝑡 , oppure considerare preferenze 'sullo stato del mondo' previste e non necessariamente osservate.

## 2.3 Variational Free Energy (VFE): ruolo nell'inferenza e calcolo

Nel problema considerato l'agente deve stimare uno stato latente 𝑠𝑡 a partire da osservazioni rumorose e parziali 𝑜𝑡 . L'inferenza bayesiana esatta richiederebbe il calcolo del posterior 𝑝(𝑠𝑡 ∣ 𝑜 𝑡 ) , che in generale può essere costoso o non disponibile in forma chiusa. In approccio variazionale, si introduce quindi una distribuzione approssimata 𝑞(𝑠𝑡 ) e la si ottimizza minimizzando la Variational Free Energy. [9]

A livello di singolo istante 𝑡 ,  data una likelihood 𝑝( 𝑜 𝑡 ∣ ∣ 𝑠 𝑡 ) e un prior 𝑝(𝑠𝑡 ) una forma operativa della VFE è:

<!-- formula-not-decoded -->

Minimizzare 𝐹𝑡 rispetto a 𝑞 equivale a rendere 𝑞(𝑠𝑡 ) una buona approssimazione del posteriore: intuitivamente, la VFE penalizza sia credenze troppo diverse dal prior sia credenze che spiegano male i dati. Nel lavoro sperimentale, 𝐹𝑡 viene calcolata e registrata come indicatore diagnostico dell'andamento dell'inferenza.

## 2.3.1 Decomposizione: accuracy vs complexity

La VFE ammette una decomposizione particolarmente utile per interpretare i risultati sperimentali:

<!-- formula-not-decoded -->

dove:

- Complexity = KL(q | p) : misura quanto il belief aggiornato q(st ) si discosta dal prior predittivo p(st ) . [10] Un aumento di complexity indica che l'agente sta 'pagando' un cambiamento di credenza rispetto a quanto previsto dal solo modello di transizione, tipicamente perché l'osservazione suggerisce un regime diverso.
- Accuracy = 𝔼q[log  p(ot ∣ s t )] :  misura  quanto  le  osservazioni  risultano  probabili  sotto  gli stati supportati da q . Quando l'osservazione è ben spiegata dal modello generativo, l'accuracy aumenta. [11]

Nel monitoring sperimentale questa decomposizione permette di distinguere due situazioni frequenti:

- (i) cambiamenti di regime 'spiegabili' dal modello, in cui l'agente aumenta complexity per adattarsi ai dati mantenendo buona accuracy;
- (ii) mismatch tra modello e dati, in cui l'accuracy cala significativamente (osservazioni poco compatibili con la likelihood), producendo un aumento della VFE anche a parità di belief .

## 2.3.2 Entropia del belief e relazione con incertezza

Accanto alla VFE, un indicatore diretto dell'incertezza dell'agente è l'entropia della distribuzione di credenza:

<!-- formula-not-decoded -->

L'entropia è massima quando 𝑞(𝑠𝑡 ) è prossima a una distribuzione uniforme (massima incertezza) e minima quando 𝑞(𝑠𝑡 ) è concentrata su un singolo stato (alta confidenza). [12] Nel contesto di osservazioni parziali, l'entropia risulta particolarmente informativa: quando il canale osservativo è ridotto o  ambiguo,  l'agente  tende  a  mantenere  belief  più  'diffusi',  con  entropia  più  elevata.  Viceversa,

quando l'informazione osservativa è ricca e diagnostica, il belief collassa più rapidamente e l'entropia diminuisce.

È importante notare che entropia e complexity non sono equivalenti: la complexity confronta 𝑞 con il prior predittivo, mentre l'entropia misura la 'dispersione interna' di 𝑞 . Tuttavia, osservare congiuntamente 𝐹𝑡 , accuracy/complexity e 𝐻(𝑞𝑡 ) consente una lettura più chiara del comportamento inferenziale durante anomalie transitorie o persistenti.

## 2.4 Expected Free Energy (EFE) per selezione dell'azione

In Active Inference la scelta dell'azione (o più in generale di una policy 𝜋 , cioè una sequenza di azioni future) viene formulata come minimizzazione della Expected Free Energy 𝐺 .[6] [7] Intuitivamente, 𝐺 valuta le conseguenze future attese di una policy bilanciando due obiettivi: (i) ottenere osservazioni 'desiderabili' (componente pragmatica/estrinseca) e (ii) acquisire informazione riducendo l'incertezza sullo stato latente (componente epistemica/intrinseca). Questa impostazione è standard nella letteratura su active inference discreto e nelle derivazioni ' process theory '.[13]

Il calcolo dell'EFE quindi comprende due componenti che sono il guadagno pragmatico e il guadagno epistemico. La funzione risulta:

<!-- formula-not-decoded -->

## 2.4.1 Componente epistemica (Information Gain)

La componente epistemica misura quanto un'azione (o policy ) è attesa ridurre l'incertezza sullo stato latente. Una formulazione comune è l'Information Gain come mutua informazione attesa tra stati e osservazioni, che può essere scritta come:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 𝑄(𝑠 ∣ 𝑎 ) è il prior predittivo sugli stati se eseguo 𝑎;
- 𝑄(𝑠 ∣ 𝑜, 𝑎 ) è il posterior dopo aver osservato 𝑜 sotto l'azione 𝑎 : 𝑄(𝑠 ∣ 𝑜, 𝑎) ∝ 𝑝(𝑜 ∣ 𝑠) 𝑄(𝑠 ∣ 𝑎) ;
- La KL quantifica quanto l'osservazione 'sposta' le credenze rispetto al prior: se l'osservazione è altamente diagnostica, la KL media cresce e quindi cresce IG.

Questa decomposizione ' epistemic value = information gain ' è presentata esplicitamente come parte della lettura intrinseca dell'EFE ed è molto usata per spiegare l'equilibrio exploration / exploitation in Active Inference. [14] [15]

Nel mio caso di studio abbiamo: azioni che aumentano l'informazione osservabile (es. attivazione del sensore temperatura) tendono ad aumentare IG perché restringono l'ambiguità tra stati latenti che, osservando solo vibrazione, possono risultare simili.

## 2.4.2 Componente pragmatica (raggiungimento preferenze)

La componente pragmatica (spesso chiamata extrinsic  value )  formalizza  il  fatto  che  l'agente  non cerca solo informazione, ma desidera anche che le osservazioni future siano compatibili con un insieme di preferenze pC (o) (omeostatiche). Una scelta operativa tipica è valutare:

<!-- formula-not-decoded -->

Questo termine 'premia' azioni che rendono più probabili osservazioni desiderate (ad esempio vicino ai set-point nominali). Nella letteratura, questa parte appare anche come extrinsic / instrumentum value o come expected utility scritta in log-probabilità, coerentemente con l'uso di preferenze probabilistiche in Active Inference.

nel setting omeostatico la preferenza può essere modellata con gaussiane su vibrazione e temperatura. In tal caso log pC (o) assegna punteggio più alto a osservazioni 'nominali' e più basso a condizioni anomale; l'azione correttiva (es. raffreddamento) avrà PG alto quando lo stato latente prevede overheating.

## 2.5 Inferenza ammortizzata: PosteriorNet

Nei modelli di Active Inference discreti, l'inferenza sullo stato latente può essere implementata in modo esplicito combinando prior e likelihood per ottenere una distribuzione 𝑞(𝑠𝑡 ) che approssima il posterior. In alternativa, al fine di ridurre il costo computazionale per step e aumentare la scalabilità, è possibile sostituire l'inferenza iterativa con una mappatura parametrica appresa che produce direttamente una stima del posterior: tale approccio è noto come amortized inference (o recognition model ) nella letteratura di Variational Inference. [16]

In questa impostazione, una rete neurale parametrizzata da 𝜃 approssima la distribuzione di credenza come:

<!-- formula-not-decoded -->

dove 𝑓 𝜃 (⋅) produce logits sugli stati discreti e 𝑥𝑡 è un vettore di feature costruito a partire dall'osservazione corrente e da variabili contestuali. L'amortizzazione sposta parte del 'lavoro' inferenziale nella fase di apprendimento: a runtime, il belief viene ottenuto tramite una singola forward-pass , in luogo di un aggiornamento bayesiano esplicito.

Questa scelta produce un'architettura ibrida: la componente inferenziale è subsimbolica (parametrica e distribuita), mentre il resto della pipeline decisionale può rimanere strutturato e ispezionabile (transizioni, likelihood e preferenze, nonché selezione d'azione tramite EFE).

## 2.5.1 - Training supervisionato in simulazione e implicazioni (limiti/assunzioni)

Nel caso in cui sia disponibile un'etichetta di riferimento per lo stato latente (ad esempio perché generato da un processo noto), la PosteriorNet può essere addestrata in modo supervisionato minimizzando una loss di c ross-entropy :

<!-- formula-not-decoded -->

dove 𝑠𝑡 ⋆ è  lo  stato  di  riferimento  al  tempo 𝑡 e 𝑞𝜃 è  la  distribuzione  predetta  dalla  rete.  In  pratica, 𝑓 𝜃 (𝑥𝑡 ) produce logits e la softmax definisce una distribuzione categorica sugli stati discreti. L'ottimizzazione avviene con metodi standard su mini-batch , eventualmente con meccanismi di replay per stabilizzare l'apprendimento.

Dal punto di vista metodologico, questo approccio consente di valutare in modo controllato l'effetto dell'ammortizzazione: l'inferenza esplicita viene sostituita da un'approssimazione neurale, mentre il criterio decisionale basato su EFE rimane invariato. [17]

È tuttavia necessario esplicitare alcune assunzioni e limiti:

1. Disponibilità delle etichette: l'addestramento supervisionato richiede accesso a 𝑠𝑡 ⋆ , condizione tipicamente soddisfatta in contesti simulativi o con etichette surrogate. In scenari reali, l'apprendimento del recognition model può richiedere formulazioni non supervisionate basate su obiettivi variazionali (ELBO/VFE).
2. Amortization gap :  una  rete  neurale  implementa  una classe funzionale vincolata e può non raggiungere l'ottimo variazionale ottenibile con un aggiornamento esplicito 'libero'; ciò può introdurre bias sistematici nel belief , soprattutto fuori distribuzione.
3. Robustezza e generalizzazione: la qualità di 𝑞𝜃 dipende dalla distribuzione dei dati utilizzata in training; cambiamenti nella dinamica o nel rumore possono degradare la stima del posterior.
4. Calibrazione dell'incertezza: modelli neurali possono produrre posteriori eccessivamente confidenti; per questo risulta utile monitorare entropia e indicatori inferenziali durante gli esperimenti.

## Capitolo 3: Metodologia e implementazione dei metodi

Il processo generativo dell'ambiente e la formalizzazione del modello (likelihood, transizioni e preferenze) sono stati introdotti nei Capitoli 1 e 2. In questo capitolo vengono descritte le implementazioni dei due approcci di inferenza confrontati, i dettagli operativi e la selezione dell'azione tramite EFE.

## 3.1 Metodo A - Inferenza esplicita del posteriore (Bayes update)

Come formalizzato nel Capitolo 2, l'inferenza sullo stato latente è trattata come un filtro bayesiano discreto: a ogni passo l'agente combina una predizione dinamica basata sulle transizioni 𝐵𝑎 e una correzione osservativa basata sulla likelihood 𝑝(𝑜𝑡 ∣ 𝑠 𝑡 ) . In questa sezione si descrive la procedura operativa adottata nell'implementazione, evidenziando le scelte necessarie per gestire osservazioni parziali, robustezza al rumore e stabilità numerica.

## 3.1.1 Calcolo likelihood e gestione osservazioni parziali

L'osservazione 𝑜𝑡 include sempre la vibrazione 𝑣𝑡 ; la temperatura 𝑇𝑡 è presente solo quando il sensore addizionale è attivo e viene letta effettivamente nello step corrente. In codice, la likelihood per ciascuno stato 𝑠 è calcolata come prodotto di termini gaussiani:

- sempre: 𝑝(𝑣𝑡 ∣ 𝑠) ;
- solo se disponibile: moltiplicazione per 𝑝(𝑇𝑡 ∣ 𝑠) .

Questo implementa direttamente l'idea di maschera osservativa 𝑚𝑡 introdotta nel Cap.2, evitando di 'inventare'  una  temperatura  quando  non  viene  misurata.  Inoltre,  per  evitare underflow numerico (moltiplicazioni di probabilità piccole) la likelihood viene limitata inferiormente a un valore 𝜀 .

Un dettaglio implementativo importante è che l'attivazione del sensore temperatura è trattata come one-shot : una volta acquisito 𝑇𝑡 , il flag viene disattivato. In questo modo la disponibilità del canale temperatura è realmente un effetto dell'azione di sensing, e non rimane 'accidentalmente' attiva per più step . La likelihood quindi ci permette di definire quanto le osservazioni risultino compatibili con

ciascuno stato latente 𝑠𝑡 secondo il modello generativo dell'agente, fornendo l'evidenza osservativa che verrà combinata con il prior predittivo per ottenere il belief aggiornato to 𝑞(𝑠𝑡 ) tramite aggiornamento bayesiano, che viene illustrato nella prossima sezione.

## 3.1.2 Aggiornamento del belief 𝑞(𝑠𝑡 )

Come discusso nel Cap.2, l'agente mantiene una distribuzione di credenza 𝑞(𝑠𝑡 ) sugli stati latenti discreti. L'aggiornamento al tempo 𝑡 è implementato come un filtro bayesiano discreto in due fasi: predizione dinamica tramite le matrici di transizione 𝐵𝑎 e correzione osservativa tramite la likelihood 𝑝(𝑜𝑡 ∣ 𝑠 𝑡 ) .

Dato il belief al tempo precedente q(st-1 ) e l'azione effettivamente eseguita nello step precedente at-1 , si calcola un prior predittivo sullo stato corrente propagando la credenza con la dinamica dipendente dall'azione:

<!-- formula-not-decoded -->

In  questa  forma, Ba è  una  matrice  colonna-stocastica  con  elementi (𝐵𝑎)𝑖𝑗 = 𝑝(𝑠𝑡 = 𝑖 ∣ 𝑠𝑡-1 = 𝑗, 𝑎 𝑡-1 = 𝑎) . Il vettore 𝑞 (𝑠𝑡 ) rappresenta quindi l'aspettativa dell'agente sullo stato prima di osservare ot .

Ricevuta  l'osservazione ot ,  si  calcola  la  likelihood  per  ciascuno  stato s , 𝐿(𝑠) = 𝑝( 𝑜𝑡 ∣ ∣ 𝑠 ) (Sez. 3.1.1).

Il belief aggiornato (posteriore approssimato) è ottenuto combinando prior predittivo e likelihood e normalizzando:

<!-- formula-not-decoded -->

In termini intuitivi, q incorpora la coerenza con la dinamica del sistema, mentre L ̃ misura la coerenza con l'osservazione: il prodotto privilegia gli stati compatibili con entrambi.

Poiché l'aggiornamento coinvolge prodotti e logaritmi, q e L ̃ vengono clampati inferiormente a un valore ε ed in seguito rinormalizzati.

Inoltre, per consentire il calcolo coerente della VFE e della decomposizione accuracy/complexity, nello step t vengono salvati:

- prior predittivo q (st ) (da Bat-1 ),
- likelihood L ̃ (s t ) ,
- belief aggiornato q(st ) .

## 3.1.3 Logging: VFE, entropia, belief per stato

Per analizzare in modo quantitativo il comportamento inferenziale del metodo A, a ogni passo t vengono registrate grandezze interne dell'agente che rendono osservabile come si aggiornano le credenze, quanta incertezza rimane, e quanto bene il modello spiega i dati.

## 3.2 Metodo B - PosteriorNet (amortized inference)

Come  introdotto  nel  Capitolo  2,  l'inferenza  ammortizzata  sostituisce  l'aggiornamento  bayesiano esplicito con una mappatura parametrica che stima direttamente la credenza sullo stato latente. In questa sezione si descrivono le scelte implementative adottate per la costruzione delle feature di input, l'addestramento supervisionato mediante replay buffer e loss di cross-entropy, e l'integrazione del belief neurale nella pipeline decisionale basata su EFE.

## 3.2.1 Feature design

Come nel Metodo A, l'agente riceve un'osservazione ot in cui la vibrazione vt è sempre disponibile, mentre la temperatura Tt è osservabile solo quando il sensore addizionale viene attivato e produce una lettura nello step corrente. Questa osservabilità parziale viene rappresentata esplicitamente tramite una maschera binaria mt ∈ {0,1} , che indica se il canale temperatura è presente.

L'input alla  PosteriorNet  è  quindi  costruito  includendo: vt , Tt (se  disponibile),  la  maschera mt ,  e l'azione precedente at-1 codificata one-hot . L'inclusione di at-1 fornisce contesto sul tipo di informazione disponibile (ad esempio se lo step corrente include la temperatura in seguito a un'azione di sensing), rendendo la stima del belief più coerente con la dinamica osservativa del sistema.

## 3.2.2 Replay buffer, loss cross-entropy, ottimizzazione

Dato che nel processo generativo simulato lo stato latente st è noto, come variabile interna del mondo, è possibile addestrare la PosteriorNet in modo supervisionato utilizzando 𝑠𝑡-1 come etichetta di riferimento. Lo stato verrà utilizzato esclusivamente come etichetta per training e valutazione, non come informazione disponibile all'agente durante la fase decisionale delle azioni.

L'addestramento viene condotto online tramite raccolta di campioni (xt , 𝑠𝑡-1 ) in un replay buffer a capacità finita. L'uso del replay buffer ha due scopi principali che sono quelli di ridurre la correlazione temporale tra campioni consecutivi; stabilizzare l'ottimizzazione mescolando esempi provenienti da regimi diversi (nominale/anomalia transitoria/anomalia persistente). [18] [19]

Data una mini-batch estratta  casualmente dal buffer, la rete produce log 𝑧𝑡 = 𝑓 𝜃 (𝑥𝑡 ) e quindi una distribuzione categorica sui tre stati tramite softmax:

<!-- formula-not-decoded -->

La  funzione  obiettivo  utilizzata  è  la cross-entropy (equivalente  alle  negative  log-likelihood  della classe corretta):

<!-- formula-not-decoded -->

L'ottimizzazione avviene con metodi standard basati su gradiente e  aggiornamenti su mini-batch . Questa procedura implementa l'idea di 'spostare' il costo dell'inferenza nella fase di apprendimento: durante l'esecuzione, il belief viene ottenuto con una singola forward-pass ,  evitando iterazioni di aggiornamento esplicite.

## 3.2.3 Uso del belief neurale nella pipeline EFE

Una volta addestrata (o durante l'addestramento online), la PosteriorNet sostituisce l'aggiornamento bayesiano esplicito producendo direttamente il belief corrente:

<!-- formula-not-decoded -->

Questo belief viene quindi utilizzato come input per tutte le quantità decisionali già definite nel Capitolo 2. In particolare:

- la predizione dinamica sotto una possibile azione 𝑎 è ottenuta applicando la matrice di transizione:

<!-- formula-not-decoded -->

- il belief entra nel calcolo delle componenti dell'Expected Free Energy (EFE), ad esempio:
1. nella  parte  epistemica  (Information  Gain),  attraverso  il  prior  predittivo  e  l'aggiornamento 'ipotetico' dopo osservazioni simulate;
2. nella parte pragmatica, attraverso la previsione degli stati futuri e la valutazione della compatibilità con le preferenze omeostatiche.

In questa impostazione, quindi, cambia solo il modo in cui si ottiene 𝑞𝑡 : nel Metodo A deriva da un update probabilistico esplicito; nel Metodo B deriva da una mappatura neurale. Il principio decisionale rimane invariato, consentendo un confronto controllato in cui si isola l'effetto dell'inferenza ammortizzata sul comportamento (azioni selezionate) e sugli indicatori inferenziali. Inoltre, vengono mantenuti i logging come nel metodo A su VFE e belief, solo come monitoring diagnostico tramite a un likelihood/transition model fissato; ciò non implica che l'agente stia aggiornando il belief con quel likelihood, ma solo che si stanno tracciando indicatori comparabili tra metodi.

## 3.3 Selezione dell'azione via EFE

Per i due metodi precedentemente illustrati (inferenza esplicita e inferenza ammortizzata) la selezione delle azioni avviene tramite la stessa funzione: a ogni istante 𝑡 , l'agente valuta un insieme di azioni candidate 𝑎 ∈ 𝐴 calcolando una stima della Expected Free Energy associata e scegliendo l'azione che minimizza tale valore.

Operativamente si adotta un orizzonte breve (tipicamente one-step), così da ottenere una decisione reattiva e comparabile tra metodi. Come visto nel paragrafo 2.4 il calcolo è effettuato con l'utilizzo del guadagno epistemico e quello pragmatico calcolato su una determinata azione.

<!-- formula-not-decoded -->

Nei paragrafi seguenti vengono descritte le procedure operative adottate per il calcolo dei due termini.

## 3.3.1 Calcolo IG

Come formalizzato nella sezione 2.4.1 data una certa azione 𝑎 , per calcolare il guadagno epistemico usiamo la formula definita come:

<!-- formula-not-decoded -->

Poiché l'aspettativa su 𝑜 non è sempre trattabile in chiuso, l'IG viene stimato con Monte Carlo [20], le fasi del calcolo possono essere divise come:

1. viene campiona uno stato 𝑠 ∼ 𝑞(𝑠𝑡+1 ∣ 𝑎) ;
2. successivamente si, campiona un'osservazione 𝑜 ∼ 𝑝(𝑜 ∣ 𝑠) ,tenendo conto della disponibilità dei canali sensoriali indotta dall'azione (es. temperatura osservabile solo se il sensing è attivato);
3. si aggiorna un posterior 'simulato' tramite Bayes usando prior e likelihood del modello generativo:

<!-- formula-not-decoded -->

4. viene calcolata la KL 𝐷KL(𝑞(𝑠 ∣ 𝑜, 𝑎) ∥ 𝑞(𝑠 ∣ 𝑎)) ;
5. Per ottenere infine la stima si effettua la media su 𝑁 campioni:

<!-- formula-not-decoded -->

Questa stima rende misurabile il valore informativo di azioni epistemiche perché tali azioni riducono l'ambiguità tra stati latenti quando l'osservazione è altrimenti poco diagnostica. Prendendo in considerazione l'azione epistemica introdotta dal caso di studio essa utilizza la temperatura per ridurre eventuali ambiguità sugli stati latenti, infatti, l'azione di sensing tende a produrre valori di 𝐼𝐺(𝑎) più elevati quando il belief è più incerto poiché l'informazione addizionale aiuta a discriminare tra regimi anomali.

## 3.3.2 Calcolo PG

La componente pragmatica valuta quanto le conseguenze attese dell'azione siano compatibili con le preferenze 𝑝𝐶(𝑜) (Cap. 2.2.3). La forma operativa è:

<!-- formula-not-decoded -->

Nel caso omeostatico, 𝑝𝐶 (𝑜) assegna punteggio alto a osservazioni vicine a valori nominali (es. vibrazione e temperatura di regime), e punteggio basso a osservazioni anomale. In pratica, invece di campionare direttamente 𝑜 , è possibile calcolare un punteggio atteso per stato e poi mediare rispetto alla predizione sugli stati.

Nel nostro caso, lo score (𝑠) corrisponde a un termine di negative cross-entropy ed è implementato nel codice tramite una espressione in forma chiusa. Senza esplicitare qui la derivazione in ciclo chiuso, è utile interpretarlo come

<!-- formula-not-decoded -->

cioè il valore atteso della log-densità di preferenza sotto la distribuzione delle osservazioni generate dallo stato 𝑠 .

In questo modo se l'agente avrà credenze forti verso l'overheating, l'azione pragmatica di cooling system avrà la meglio sulle altre due azioni perché le osservazioni future che saranno osservate apparterranno ad uno stato normale.

## 3.3.3 Softmax/argmax

Una volta che viene calcolato l'EFE su tutte le azioni candidate 𝑎 , l'agente costruisce una distribuzione tramite una softmax con una precisione 𝛾 :

<!-- formula-not-decoded -->

𝛾 controlla quanto la politica sia impattante: valori alti rendono la softmax più concentrata sull'azione migliore.

Infine, per mantenere un comportamento deterministico e facilmente confrontabile, l'azione viene selezionata tramite:

<!-- formula-not-decoded -->

dove 𝑎𝑡 equivale all'azione con probabilità maggiore.

Per analizzare il comportamento di ciascuna azione vengono registrati i valori EFE di ciascuna di esse su W&amp;B.

## 3.4 Vincoli fisici separati dal criterio decisionale

All'interno  del  mondo  è  presente  un  vincolo  fisico  che  influisce  sull'esecuzione  delle  azioni,  in quanto, la batteria del sistema diminuisce nel tempo e limita l'utilizzo di azioni che richiedono energia. Questo vincolo è una semplice soglia ed è modellato come parte del processo generativo e non come parte del criterio di decisionale, al fine di mantenere confrontabili i metodi sul piano inferenziale/decisionale. La batteria nel mondo sarebbe il vincolo di risorsa che permette al drone di 'sopravvivere' nel tempo. Il livello di carica viene aggiornato dopo ogni step in base al costo associato ad ogni azione eseguita. azione nulla ( 𝑎 = 0 ) ha costo pari a 0, poiché non attiva sensori aggiuntivi e non produce interventi sull'ambiente. Al contrario, l'azione epistemica ( 𝑎 = 1 ), che consiste nell'attivazione del sensore di temperatura, comporta una riduzione della batteria pari allo 0,01% per attivazione, mentre l'azione pragmatica ( 𝑎 = 3 ), corrispondente all'attivazione del cooling system, produce una riduzione pari allo 0,4%. I valori sono stati mantenuti volutamente contenuti al fine di prolungare la durata della simulazione e consentire l'osservazione di più episodi e transizioni tra regimi operativi.

## Capitolo 4: Metodi sperimentali e risultati

Questo capitolo presenta il protocollo sperimentale adottato e i risultati ottenuto nel confronto tra i tre approcci descritti nei capitoli precedenti. L'obbiettivo è quello di rispondere alle domande che sono state definite nella sezione 1.6 valutando: la gestione dell'incertezza, la qualità della selezione d'azione e il trade-off tra prestazioni e costo.

## 4.1 Setup sperimentale

L'esperimento è eseguito su una sequenza a tempo discreto di durata 𝑇 , durante la quale l'ambiente produce osservazioni 𝑜𝑡 e l'agente seleziona azioni 𝑎𝑡 secondo uno dei metodi in confronto. Le osservazioni includono sempre la vibrazione 𝑣 𝑡 e, quando il sensore addizionale è attivato, anche la temperatura 𝑇𝑡 . I due grafici sottostanti mostrano un esempio di traccia temporale delle osservazioni generate dal processo, evidenziando la variabilità nominale e le deviazioni indotte dalle anomalie evidenziando anche come il sensore della temperatura è attivo solo per uno step quando attivato.

Per rendere il confronto riproducibile e controllato, le anomalie vengono introdotte a intervalli regolari: ogni 𝑁 = 100 step viene iniettato un evento anomalo, scelto tra perturbation (transitoria) e overheating (persistente), secondo la procedura descritta nel Capitolo 1.

<!-- image -->

## 4.2 Analisi congiunta: belief, VFE e decomposizione accuracy/complexity

In questa sezione l'analisi quantitativa tramite VFE, accuracy e complexity viene condotta sui soli metodi  che  mantengono  una  rappresentazione  probabilistica  dello  stato  latente,  ovvero  inferenza esplicita e amortized inference. Il metodo a soglie non è disposto di una distribuzione di credenze su uno stato e di conseguenza non sono calcolabili valori come belief e VFE.

Come specifica nel capito 3 i due metodi calcolano la belief in due modi differenti: il primo si avvale di un Bayes update mentre il secondo metodo di una rete neurale. Per dimostrare la correlazione tra VFE e belief utilizzeremo il primo metodo che è l'equivalente di usare il secondo in questa analisi.

Analizzando la figura telemetrie dello scorso paragrafo e prendendo come esempio un'anomalia, in questo caso una perturbazione, possiamo osservare nel grafico sottostante come le credenze sugli stati nel momento che viene iniettata un'anomalia si dividono tra perturbation e overheaiting mentre le credenze sullo stato nominale diventano scendono drasticamente.

<!-- image -->

Come definito nel capitolo 2 la VFE è una misura di 'quanto bene' l'agente sta spiegando l'osservazione con il suo modello, tenendo conto anche di quanto deve cambiare idea. Nel nostro caso di studio torna utile l'utilizzo della decomposizione della VFE e quindi dell'utilizzo dei parametri di accuracy

e  complexity per spiegare le credenze sugli stati. Di seguito sono analizzati i due grafici dei due componenti.

Possiamo osservare come nel momento che viene iniettata l'anomalia si ha un aumento nella complessità che indica che la credenza sullo stato si discosta molto dal prior predittivo calcolato allo step precedente. È interessante inoltre notare come dopo il passaggio dell'anomalia, la complexity risalga quando il sistema torna allo stato nominale (molto comune nella perturbation).

<!-- image -->

Per quanto riguarda l'accuracy essa al contrario della complexity ha un drastico calo quando viene iniettata l'anomalia in quanto si ha un mismatch tra i dati osservati e ciò che il modello generativo si aspetta di osservare nei diversi stati.

<!-- image -->

In entrambe le metriche si osserva un miglioramento quando viene scelta l'azione epistemica perché permette di spiegare lo stato grazie alla misurazione della temperatura e quindi di stabilizzare i due valori.

Grazie a questo confronto ora siamo in grado di rispondere alla prima delle domande che ci siamo posti nella sezione 1.6 : Quanto i tre approcci differiscono nella gestione dell'incertezza e nella stabilità del comportamento durante anomalie transitorie e persistenti?

Nel metodo simbolico a soglie la selezione delle azioni dipende esclusivamente da regole su osservazioni grezze (ad esempio soglie sulla vibrazione). In assenza di una belief sugli stati, il sistema non può:

- misurare esplicitamente l'incertezza sullo stato,
- distinguere in modo sistematico tra fluttuazione e anomalia persistente,
- legare l'attivazione del sensing a una condizione di ambiguità interna.

Di conseguenza, a parità di soglie, l'azione epistemica può essere attivata anche quando non necessaria (per esempio in presenza di rumore o oscillazioni non informative), mentre nei metodi probabilistici l'attivazione risulta interpretabile come risposta a una fase di incertezza misurabile (belief diffusa / complexity alta / accuracy bassa).

## 4.3 Analisi EFE e selezione dell'azione

In questo paragrafo vengo analizzati i valori della softmax delle tre azioni del sistema: a ∈ {0,1,3}. L'analisi viene condotta sul medesimo caso di studio introdotto nei paragrafi precedenti, mettendo in relazione l'andamento delle probabilità sulle azioni con i grafici già discussi per il belief e per gli indicatori inferenziali (VFE, accuracy, complexity). In particolare, l'obiettivo è mostrare come, nelle fasi in cui l'agente si trova in una condizione di ambiguità (belief più diffuso e complessità più elevata), aumenti la probabilità assegnata all'azione epistemica 𝑎 = 1 , mentre in presenza di evidenza compatibile  con  una  condizione  persistente  (overheating)  la  distribuzione  tenda  a  concentrarsi sull'azione correttiva 𝑎 = 3 .

<!-- image -->

Nel grafico di 𝑝(𝑎𝑡 = 0) si osserva che la probabilità assegnata all'azione neutra rimane elevata per gran parte dell'episodio, tipicamente nell'intervallo 0.8 -0.9 . Questo comportamento è coerente con il ruolo dell'azione 𝑎 = 0 come default conservativo: in assenza di segnali informativi o pragmatici sufficienti, l'agente tende a evitare operazioni aggiuntive (sensing e cooling), privilegiando la stabilità operativa e il contenimento dei costi.

<!-- image -->

Nel grafico di 𝑝(𝑎𝑡 = 1) si osserva che la probabilità associata all'azione epistemica (attivazione del sensore temperatura) rimane generalmente contenuta lungo l'episodio, con un valore di fondo tipicamente nell'intervallo 0.1 -0.2 . Questo comportamento è coerente con la natura di 𝑎 = 1 : il sensing addizionale viene selezionato solo quando produce un guadagno informativo atteso sufficiente da giustificarne il costo (energetico e/o operativo).

Sono però presenti picchi localizzati, che indicano istanti in cui l'azione 𝑎 = 1 diventa improvvisamente competitiva rispetto all'azione 𝑎 = 0 . Essi sono interpretabili come fasi in cui l'agente rileva una condizione di ambiguità osservativa cioè quando la sola vibrazione non è sufficiente per discriminare gli stati latenti.

<!-- image -->

Il grafico di 𝑝(𝑎𝑡 = 3) , associato all'azione pragmatica di attivazione del sistema di raffreddamento, mostra un comportamento nettamente diverso rispetto a 𝑎0 e 𝑎1 , per gran parte dell'episodio la probabilità resta prossima allo zero, mentre compaiono picchi isolati e ben localizzati.

Questa dinamica è coerente con il ruolo di 𝑎 = 3 nel caso di studio: l'azione di cooling è un intervento costoso, quindi, viene selezionata solo quando l'agente stima che lo stato latente più plausibile corrisponda a una condizione persistente che richiede correzione, nel caso di studio questo corrisponde all'overheating.

Grazie all'analisi fatta ora possiamo rispondere alla seconda domanda: L'approccio con EFE migliora la selezione delle azioni rispetto ad un approccio baseline con treshold?

Per rispondere alla domanda confrontiamo la scelta dell'azione dell'approccio probabilistico rispetto alla baseline simbolica a soglie, dove l'attivazione del sensore è legata rigidamente al superamento di una regola su 𝑣𝑡 , nei metodi EFE l'azione epistemica emerge solo quando è giustificata da una condizione interna misurabile di incertezza.

<!-- image -->

Possiamo quindi affermare che l'approccio probabilistico con EFE risulta quindi più interpretabile come risposta ad ambiguità e più efficace in termini di costo.

## 4.4 PosteriorNet e confronto qualitativo con inferenza esplicita

Per il metodo B, l'inferenza sullo stato latente non è ottenuta tramite aggiornamento bayesiano esplicito, ma tramite una stima parametrica 𝑞𝜃(𝑠𝑡 ∣ 𝑜 𝑡 , 𝑎 𝑡-1 ) . Di conseguenza, a differenza dell'inferenza esplicita, la qualità del belief dipende anche dalla fase di apprendimento della rete. Il grafico dell'entropia 𝐻(𝑞𝑡 ) evidenzia chiaramente questa dinamica: nelle prime fasi l'agente neurale tende a produrre belief poco calibrati e instabili, con valori di entropia più elevati e oscillazioni più ampie, perché la rete non ha ancora appreso una mappatura affidabile dalle osservazioni allo stato latente.

In questa fase iniziale l'incertezza riflette un errore di stima legato all'addestramento ancora incompleto e non dovuto all'ambiguità intrinseca delle osservazioni o alla dinamica del processo generativo.

<!-- image -->

Con l'aumentare delle interazioni e l'aggiornamento dei pesi, la PosteriorNet converge verso una rappresentazione più coerente del posteriore: l'entropia del belief si stabilizza e mostra un andamento più comparabile a quello del Metodo A, in cui 𝐻(𝑞𝑡 ) cresce tipicamente in corrispondenza di osservazioni ambigue o anomalie (fase di disambiguazione) e decresce quando l'evidenza osservativa permette un collasso del belief su uno stato specifico. In altre parole, dopo la fase di assestamento, l'inferenza ammortizzata riesce a riprodurre un profilo di incertezza simile all'inferenza esplicita: l'entropia diventa interpretabile come misura di ambiguità del regime operativo e non come instabilità del modello.

Ora che abbiamo una prospettiva sulla PosteriorNet possiamo rispondere all'ultima domanda: La PosteriorNet  consente  di  ridurre  costo  computazionale  mantenendo  prestazioni  comparabili  rispetto all'inferenza esplicita?

Una volta stabilizzata, la PosteriorNet fornisce prestazioni inferenziali qualitativamente comparabili all'aggiornamento esplicito, pur sostituendo l'inferenza con una stima diretta del belief. La differenza principale rimane la presenza di una fase transiente di apprendimento, durante la quale l'incertezza misurata può risultare sovrastimata o meno informativa.

## Conclusioni

In relazione alle domande di ricerca poste nel Capitolo 1, i risultati raggiunti permettono di fare tre considerazioni principali. Per quanto riguarda le differenze tra i metodi nel modo in cui affrontano l'incertezza e la stabilità di comportamento, la soluzione deterministica a soglie funziona bene e in modo prevedibile in condizioni nominali, ma tende a peggiorare nei casi in cui il rumore è elevato e l'osservabilità è parziale, mentre l'approccio basato su EFE si comporta in modo più robusto nei casi in cui è presente ambiguità, visto che la scelta dell'azione dipende anche dallo stato interno di credenza. Per quanto riguarda il confronto tra selezione di azione basata su EFE e baseline a soglie, l'EFE ha il vantaggio di poter attivare azioni epistemiche quando esse sono necessarie (per esempio, per ridurre l'incertezza) e di preferire azioni pragmatiche quando l'evidenza verso uno stato anomalo persistente è sufficientemente forte. Infine, per quanto riguarda l'inferenza ammortizzata, la PosteriorNet può ridurre il costo computazionale a runtime ed essere competitiva dopo il training, ma necessita di una fase di addestramento e di un'accurata valutazione della sua capacità di generalizzazione, l'inferenza esplicita invece è più trasparente e controllabile, mentre la strada da percorrere sembra essere una soluzione ibrida in cui c'è ancora una parte esplicita e alcune parti sono   apprese.

Nel complesso, l'analisi suggerisce che un approccio simbolico deterministico può essere una scelta valida nei sistemi decisionali in scenari IoT per semplicità e costo computazionale contenuto, mentre l'impiego di criteri basati su EFE risulta particolarmente indicato quando l'ambiente è rumoroso e l'informazione è incompleta.

Infine, pur essendo stato presentato in un contesto UA V , questo impianto è generalizzabile a numerosi scenari IoT in cui un agente deve selezionare azioni sotto incertezza, soprattutto quando il processo decisionale dipende da sensori imperfetti e da vincoli di risorsa.

## Bibliografia

- [1] Cai, G., Chen, B. M., &amp; Lee, T. H., Unmanned Rotorcraft Systems. Springer, 2011.
- [2] Sajid, N., Ball, P. J., Parr, T., &amp; Friston, K. J., 'Active Inference: Demystified and Compared,' Neural Computation , 2021.
- [3] Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V ., &amp; Friston, K., 'Active inference on discrete state-spaces: A synthesis,' Journal of Mathematical Psychology , 2020.
- [4] Kaelbling, L. P., Littman, M. L., &amp; Cassandra, A. R., 'Planning and Acting in Partially ObservableStochastic Domains,' Artificial Intelligence , 1998.
- [5] Sutton, R. S., &amp; Barto, A. G., Reinforcement Learning: An Introduction , 2nd ed., MIT Press, 2018.
- [6] Friston, K., 'The free-energy principle: a unified brain theory?' Nature Reviews Neuroscience , 2010.
- [7] Parr, T., Pezzulo, G., &amp; Friston, K. J., Active Inference: The Free Energy Principle in Mind, Brain, and Behavior , The MIT Press, 2022.
- [8] Weights &amp; Biases, 'Experiments overview,' Documentation, consultata il 03/02/2026.
- [9] Blei, D. M., Kucukelbir, A., &amp; McAuliffe, J. D., 'Variational Inference: A Review for Statisticians,' Journal of the American Statistical Association , 2017.
- [10] Kullback, S., &amp; Leibler, R. A., 'On Information and Sufficiency,' The Annals of Mathematical Statistics , 1951.
- [11] Bogacz, R., 'A tutorial on the free-energy framework for modelling perception and learning,' Journal of Mathematical Psychology , 2017.
- [12] Shannon, C. E., 'A Mathematical Theory of Communication,' Bell System Technical Journal .
- [13] Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., &amp; Pezzulo, G., 'Active inference: A process theory,' Neural Computation , 2017
- [14] Friston, K. J., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., &amp; Pezzulo, G., 'Active inference and epistemic value,' Cognitive Neuroscience , 2015.

- [15] Cover, T. M., &amp; Thomas, J. A., Elements of Information Theory , 2nd ed., Wiley-Interscience, 2006.
- [16] Kingma, D. P., &amp; Welling, M., 'Auto-Encoding Variational Bayes,' 2013.
- [17] Goodfellow, I., Bengio, Y., &amp; Courville, A., Deep Learning , The MIT Press, 2016.
- [18] Lin, L.-J., 'Self-improving reactive agents based on reinforcement learning, planning and teaching,' Machine Learning , 1992.
- [19] Mnih, V., et al., 'Human-level control through deep reinforcement learning,' Nature , 2015.
- [20] Rezende, D. J., Mohamed, S., &amp; Wierstra, D., 'Stochastic Backpropagation and Approximate Inference in Deep Generative Models,' 2014.

- [21] Stoian, V., et al., "Active Inference for Autonomous UA V Inspection", 2023.
