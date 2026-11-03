(()=>{const $=id=>document.getElementById(id),main=document.querySelector('main.main'),grid=document.querySelector('.grid'),rows=$('dataChart')?.closest('.rowcharts'),bottom=document.querySelector('.bottomgrid'),tabs=$('modelTabs'),aside=document.querySelector('aside'),method=$('methodSelect'),algo=$('algorithm');if(!main||!grid||!rows||!bottom||!method||!algo)return;
const labels=new Map([...method.querySelectorAll('option')].map(o=>[o.value,o.value==='logistic'?'Regressione logistica binaria':o.textContent.trim()]));const groups={classification:['logistic','softmax','perceptron','knn','svmrbf','svmpoly','tree','forest','boost','naive','mlp'],regression:['linear','ridge','lasso','elastic','boost','mlp'],clustering:['kmeans','dbscan','gmm'],other:['sgd','momentum','nesterov','adagrad','rmsprop','adam','adamw']};
const names={classification:'Classificazione',regression:'Regressione',clustering:'Clustering',other:'Ottimizzatori'};let activeTask='classification';function renderOptions(sel,task){sel.replaceChildren();let group=document.createElement('optgroup');group.label=names[task];for(let id of groups[task])if(labels.has(id)){let o=document.createElement('option');o.value=id;o.textContent=labels.get(id);group.append(o)}sel.append(group)}function triggerMethod(id){method.value=id;algo.value=id;method.dispatchEvent(new Event('change',{bubbles:true}))}
const trainingToolbar=document.createElement('section');trainingToolbar.id='trainingToolbar';trainingToolbar.className='training-toolbar';trainingToolbar.setAttribute('aria-label','Controlli simulazione');trainingToolbar.innerHTML='<button type="button" id="trainingPause" class="secondary" disabled>⏸ Pausa</button><span class="training-toolbar-hint">Puoi mettere in pausa e riprendere senza perdere i progressi.</span>';tabs.after(trainingToolbar);const pauseButton=trainingToolbar.querySelector('#trainingPause'),trainButton=$('train'),stepButton=$('step'),progressBar=$('progress')?.parentElement,statusText=$('status'),toolbarHint=trainingToolbar.querySelector('.training-toolbar-hint');if(trainButton&&stepButton&&progressBar&&statusText){trainButton.textContent='▶ Avvia / Continua';trainButton.removeAttribute('style');stepButton.removeAttribute('style');progressBar.classList.add('training-toolbar-progress');statusText.classList.add('training-toolbar-status');trainingToolbar.append(trainButton,stepButton,pauseButton,progressBar,statusText,toolbarHint);}
const modelButtons=[...tabs.querySelectorAll('[data-model]')];tabs.replaceChildren();for(const task of Object.keys(names)){let btn=document.createElement('button');btn.dataset.model=task;btn.textContent=names[task];tabs.append(btn);btn.addEventListener('click',()=>selectTask(task))}
function selectTask(task){activeTask=task;tabs.querySelectorAll('[data-model]').forEach(b=>b.classList.toggle('active',b.dataset.model===task));renderOptions(method,task);renderOptions(algo,task);let preferred={classification:'logistic',regression:'ridge',clustering:'kmeans',other:'adam'}[task],current=method.value;let chosen=groups[task].includes(current)?current:preferred;if(task==='regression'&&!$('dataset').value.startsWith('regression'))$('dataset').value='regression';else if(task!=='regression'&&$('dataset').value.startsWith('regression'))$('dataset').value='blobs';const ds=$('dataset');[...ds.options].forEach(o=>o.hidden=(task==='regression'?!o.value.startsWith('regression'):o.value.startsWith('regression')));ds.querySelectorAll('optgroup').forEach(group=>group.hidden=![...group.querySelectorAll('option')].some(option=>!option.hidden));triggerMethod(chosen);ds.dispatchEvent(new Event('change',{bubbles:true}));updateView();}
const algorithmCapabilityNotice=document.createElement('div');algorithmCapabilityNotice.className='note';algorithmCapabilityNotice.id='algorithmCapabilityNotice';algo.after(algorithmCapabilityNotice);
function updateAlgorithmCapabilities(){
 const classCount=Number($('nClasses').textContent)||0,logistic=method.querySelector('option[value="logistic"]'),algorithmLogistic=algo.querySelector('option[value="logistic"]'),classificationActive=activeTask==='classification';
 if(logistic){logistic.disabled=classCount!==2;logistic.textContent=labels.get('logistic');}
 if(algorithmLogistic){algorithmLogistic.disabled=classCount!==2;algorithmLogistic.textContent=labels.get('logistic');}
 if(classificationActive&&classCount>2&&method.value==='logistic'&&method.querySelector('option[value="softmax"]'))triggerMethod('softmax');
 const selectedMethod=method.value,needsClasses=classificationActive||activeTask==='other',singleClassUnsupported=needsClasses&&classCount<2,unsupportedLogistic=selectedMethod==='logistic'&&classCount!==2;
 if(selectedMethod==='logistic')$('formula').textContent='p(y=1|x) = σ((w₁−w₀)ᵀx + b₁−b₀)';
 const blockingMessage=singleClassUnsupported?'Un solo target non definisce un problema di classificazione supervisionata. Passa a Clustering per esplorare questi dati.':unsupportedLogistic?'La regressione logistica è binaria; per più di due classi usa Softmax.':'';
 algorithmCapabilityNotice.textContent=blockingMessage||(classificationActive&&classCount>2?'Dataset multiclasse: Softmax gestisce tutte le classi; la regressione logistica binaria è disponibile solo con due.':'');
 $('train').disabled=!!blockingMessage;$('step').disabled=!!blockingMessage;
 if(blockingMessage)$('status').textContent=blockingMessage;
 else if($('status').textContent===algorithmCapabilityNotice.dataset.lastMessage)$('status').textContent='Pronto per l’addestramento';
 algorithmCapabilityNotice.dataset.lastMessage=blockingMessage;
}
$('dataset').addEventListener('change',updateAlgorithmCapabilities);$('classCount')?.addEventListener('change',updateAlgorithmCapabilities);
// Keep editable method parameters in configuration; formulas and evaluations have a dedicated page.
const config=algo.closest('.panel'),datasetPanel=$('dataset').closest('.panel'),lossPanel=$('lossChart').closest('.panel'),weightsPanel=$('weightsPanel'),methodLab=$('methodLab'),parameterSection=methodLab?.querySelector('#methodParams')?.parentElement;if(parameterSection){const parameterRow=parameterSection.parentElement;config.querySelector('.pb').insertBefore(parameterSection,$('split').previousElementSibling);parameterRow.replaceWith(parameterRow.firstElementChild);}config.addEventListener('change',event=>{const target=event.target;if(target instanceof HTMLInputElement&&target.dataset.param==='hidden'&&supportsNN())window.dispatchEvent(new Event('mlp-config-change'))});const nnSettings=document.createElement('div');nnSettings.id='nnConfigSettings';nnSettings.innerHTML='<div class="twocol"><div><label for="nnOutput">Output layer</label><select id="nnOutput"><option value="auto" selected>Automatico dal dataset</option><option value="binary">Sigmoid · binario</option><option value="softmax">Softmax · multiclasse</option></select></div></div><div class="note">I neuroni nascosti e il tipo di output definiscono l’architettura iniziale della rete.</div>';config.querySelector('.pb').append(nnSettings);nnSettings.querySelector('#nnOutput').addEventListener('change',()=>window.dispatchEvent(new Event('mlp-config-change')));
const tutorial=document.createElement('section');tutorial.className='panel algorithm-tutorial';tutorial.id='algorithmTutorial';tutorial.dataset.area='tutorial';tutorial.innerHTML='<div class="ph">Guida didattica · percorso e calcoli passo passo</div><div class="pb"><div class="tutorial-intro"><div><div class="tutorial-kicker">GUIDA ALL’ALGORITMO</div><h2 id="tutorialTitle"></h2><p id="tutorialPurpose" class="muted"></p></div><div class="tutorial-example"><strong>Esempio pratico con i dati selezionati</strong><p id="tutorialExample"></p></div></div><div class="tutorial-progress" id="tutorialProgress" aria-label="Avanzamento del percorso"></div><article class="tutorial-step"><div class="tutorial-step-count" id="tutorialStepCount"></div><h3 id="tutorialStepTitle"></h3><p id="tutorialStepBody"></p></article><div class="tutorial-actions"><button type="button" class="secondary" id="tutorialPrevious">← Indietro</button><button type="button" id="tutorialNext">Avanti →</button><button type="button" class="secondary" id="tutorialRunStep">▶ Esegui un passo reale nel playground</button></div><div class="tutorial-result" id="tutorialRunResult" aria-live="polite"></div><div class="small muted tutorial-tip">La traccia numerica mostra i calcoli sui dati correnti; usa “Esegui un passo reale” per applicare un aggiornamento al modello e osservare i valori cambiare.</div><section class="tutorial-calculations" aria-labelledby="tutorialCalculationTitle"><h2 id="tutorialCalculationTitle">Calcolo numerico passo passo</h2><p class="small muted" id="tutorialTraceStatus"></p><ol id="tutorialTraceSteps" class="tutorial-trace-steps"></ol></section></div>';
const tutorialControls=document.createElement('section');tutorialControls.id='tutorialControls';tutorialControls.className='tutorial-controls';tutorialControls.innerHTML='<h2>Scegli algoritmo e parametri</h2><p class="small muted">Queste impostazioni si applicano ai calcoli e alla simulazione della guida.</p>';const tutorialControlBody=document.createElement('div');tutorialControlBody.className='tutorial-controls-grid';const selectorColumn=methodLab?.querySelector('#methodSelect')?.closest('div');if(selectorColumn)tutorialControlBody.append(selectorColumn);if(parameterSection)tutorialControlBody.append(parameterSection);tutorialControls.append(tutorialControlBody);tutorial.querySelector('.pb').prepend(tutorialControls);
if(methodLab){methodLab.dataset.area='method';methodLab.classList.add('method-page');main.insertBefore(methodLab,grid);}main.insertBefore(tutorial,grid);const traceList=$('tutorialTraceSteps'),visualization=document.createElement('section');visualization.className='tutorial-visualization';visualization.id='tutorialVisualization';visualization.innerHTML='<h3 id="tutorialVisualizationTitle">Visualizzazione interattiva del modello</h3><p class="small muted" id="tutorialVisualizationNote"></p><label for="tutorialSampleIndex">Campione da esplorare</label><input id="tutorialSampleIndex" type="range" min="1" max="1" value="1"><div class="small muted" id="tutorialSelectedSample"></div><canvas id="tutorialVisualizationChart" aria-label="Visualizzazione interattiva di dati e modello"></canvas><div class="tablewrap tutorial-distance-table"><table class="data"><thead id="tutorialDistanceHead"></thead><tbody id="tutorialDistanceRows"></tbody></table></div><div class="tablewrap tutorial-algorithm-detail-wrap"><h4 id="tutorialAlgorithmDetailTitle"></h4><table class="data"><tbody id="tutorialAlgorithmDetails"></tbody></table></div>';traceList?.before(visualization);const neuralVisualization=document.createElement('section');neuralVisualization.className='tutorial-visualization tutorial-neural-network';neuralVisualization.id='tutorialNeuralNetwork';neuralVisualization.hidden=true;neuralVisualization.innerHTML='<h3>Rete neurale · neuroni e pesi per passo</h3><p class="small muted" id="tutorialNeuralNetworkNote"></p><canvas id="tutorialNeuralNetworkChart" aria-label="Rete neurale con attivazioni dei neuroni e pesi aggiornati passo passo"></canvas><div class="small muted">Linee blu/arancioni = pesi correnti W⁽ˡ⁾ (spessore proporzionale al valore assoluto); linea grigia tratteggiata = peso al passo precedente. I nodi mostrano l’attivazione corrente.</div>';visualization.after(neuralVisualization);let tutorialPlotState=null;$('tutorialSampleIndex').addEventListener('input',renderTutorialVisualization);$('tutorialDistanceRows').addEventListener('click',event=>{const row=event.target.closest('tr[data-sample-index]');if(!row)return;$('tutorialSampleIndex').value=String(Number(row.dataset.sampleIndex)+1);renderTutorialVisualization();});$('tutorialVisualizationChart').addEventListener('click',event=>{if(!tutorialPlotState)return;const rect=event.currentTarget.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,index=tutorialPlotState.nearest(x,y);if(index>=0){$('tutorialSampleIndex').value=String(index+1);renderTutorialVisualization();}});
const oldSampleSelector=$('tutorialSampleIndex'),sampleSelector=oldSampleSelector.cloneNode();oldSampleSelector.replaceWith(sampleSelector);sampleSelector.type='text';sampleSelector.removeAttribute('min');sampleSelector.removeAttribute('max');sampleSelector.removeAttribute('list');sampleSelector.setAttribute('inputmode','numeric');sampleSelector.setAttribute('autocomplete','off');sampleSelector.setAttribute('placeholder','Scrivi o scegli un numero');sampleSelector.setAttribute('aria-describedby','tutorialSelectedSample');sampleSelector.setAttribute('role','combobox');sampleSelector.setAttribute('aria-autocomplete','list');sampleSelector.setAttribute('aria-controls','tutorialSampleSuggestions');sampleSelector.setAttribute('aria-expanded','false');const sampleSearch=document.createElement('div');sampleSearch.className='tutorial-sample-search';sampleSelector.before(sampleSearch);sampleSearch.append(sampleSelector);const sampleSuggestions=document.createElement('div');sampleSuggestions.id='tutorialSampleSuggestions';sampleSuggestions.className='tutorial-sample-suggestions';sampleSuggestions.setAttribute('role','listbox');sampleSuggestions.hidden=true;sampleSelector.after(sampleSuggestions);tutorial.querySelector('label[for="tutorialSampleIndex"]').textContent='Cerca un campione per numero';let tutorialSampleEntries=[],tutorialHoveredSampleIndex=null,sampleSuggestionsOpen=false;
function updateTutorialSampleSuggestions(){const query=sampleSelector.value.trim(),matches=tutorialSampleEntries.map((entry,index)=>({entry,index})).filter(({index})=>!query||String(index+1).startsWith(query)).slice(0,12);sampleSuggestions.replaceChildren(...matches.map(({entry,index})=>{const option=document.createElement('button');option.type='button';option.className='tutorial-sample-suggestion';option.setAttribute('role','option');option.dataset.sampleIndex=String(index);option.innerHTML=`<strong>Campione ${index+1}</strong><span>target ${Number(entry.y).toFixed(4)} · x₁ ${Number(entry.x[0]).toFixed(3)} · x₂ ${Number(entry.x[1]||0).toFixed(3)}</span>`;option.addEventListener('pointerenter',()=>{tutorialHoveredSampleIndex=index;renderTutorialVisualization();});option.addEventListener('pointerleave',()=>{if(tutorialHoveredSampleIndex===index){tutorialHoveredSampleIndex=null;renderTutorialVisualization();}});option.addEventListener('click',()=>{sampleSelector.value=String(index+1);sampleSelector.removeAttribute('aria-invalid');tutorialHoveredSampleIndex=null;sampleSuggestionsOpen=false;sampleSuggestions.hidden=true;sampleSelector.setAttribute('aria-expanded','false');renderTutorialVisualization();sampleSelector.focus();});return option;}));sampleSuggestions.hidden=!sampleSuggestionsOpen||!matches.length;sampleSelector.setAttribute('aria-expanded',String(!sampleSuggestions.hidden));}
sampleSelector.addEventListener('focus',()=>{sampleSuggestionsOpen=true;updateTutorialSampleSuggestions();});sampleSelector.addEventListener('input',()=>{sampleSuggestionsOpen=true;tutorialHoveredSampleIndex=null;updateTutorialSampleSuggestions();});
sampleSelector.addEventListener('change',()=>{const value=Number(sampleSelector.value),max=Number(sampleSelector.dataset.sampleCount||0);if(Number.isInteger(value)&&value>=1&&value<=max){sampleSelector.removeAttribute('aria-invalid');tutorialHoveredSampleIndex=null;sampleSuggestionsOpen=false;sampleSuggestions.hidden=true;sampleSelector.setAttribute('aria-expanded','false');renderTutorialVisualization();}else{sampleSelector.setAttribute('aria-invalid','true');$('tutorialSelectedSample').textContent=`Inserisci o scegli un numero di campione da 1 a ${max}.`;}});
sampleSelector.addEventListener('keydown',event=>{const options=[...sampleSuggestions.querySelectorAll('.tutorial-sample-suggestion')],active=sampleSuggestions.querySelector('.tutorial-sample-suggestion[aria-selected="true"]'),activeIndex=options.indexOf(active);if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();sampleSuggestionsOpen=true;updateTutorialSampleSuggestions();const refreshed=[...sampleSuggestions.querySelectorAll('.tutorial-sample-suggestion')];if(!refreshed.length)return;const next=(activeIndex+(event.key==='ArrowDown'?1:-1)+refreshed.length)%refreshed.length;refreshed.forEach((option,index)=>option.setAttribute('aria-selected',String(index===next)));tutorialHoveredSampleIndex=Number(refreshed[next].dataset.sampleIndex);renderTutorialVisualization();}else if(event.key==='Enter'&&tutorialHoveredSampleIndex!==null){event.preventDefault();sampleSuggestions.querySelector(`[data-sample-index="${tutorialHoveredSampleIndex}"]`)?.click();}else if(event.key==='Escape'){sampleSuggestionsOpen=false;tutorialHoveredSampleIndex=null;updateTutorialSampleSuggestions();renderTutorialVisualization();}});
sampleSelector.addEventListener('blur',()=>setTimeout(()=>{if(!sampleSuggestions.matches(':hover')){sampleSuggestionsOpen=false;tutorialHoveredSampleIndex=null;updateTutorialSampleSuggestions();renderTutorialVisualization();}},120));
const dataChart=rows.children[0],hist=rows.querySelector('#histPanel'),sample=bottom.children[0],prediction=bottom.children[1],inputs=bottom.children[2],results=methodLab?.querySelector('#methodMetrics')?.closest('.panel');
if(results){
 results.dataset.area='tutorial';
 const tutorialBody=tutorial.querySelector('.pb'),learningLayout=document.createElement('div'),tutorialMain=document.createElement('div'),symbolSidebar=document.createElement('aside');
 learningLayout.className='tutorial-learning-layout';
 tutorialMain.className='tutorial-main-column';
 symbolSidebar.className='tutorial-symbol-sidebar';
 symbolSidebar.id='tutorialSymbolSidebar';
 symbolSidebar.innerHTML='<h2>Significato dei simboli</h2><p class="small muted">I simboli usati nel percorso dell’algoritmo selezionato.</p><ul id="tutorialSymbolLegend" class="method-symbol-legend"></ul>';
 learningLayout.append(tutorialMain,symbolSidebar);
 tutorialBody.append(learningLayout);
 [...tutorialBody.children].filter(element=>element!==tutorialControls&&element!==learningLayout).forEach(element=>tutorialMain.append(element));
 $('tutorialRunStep').closest('.tutorial-actions').after(results);
}
const learningData=document.createElement('section');learningData.className='panel';learningData.id='learningData';learningData.innerHTML='<div class="ph">Dati apprendimento · valori e parametri aggiornati a ogni epoca</div><div class="pb"><div class="learninggrid" id="learningGrid"></div></div>';const learningGrid=learningData.querySelector('#learningGrid');main.insertBefore(learningData,rows);const networkStructurePanel=document.createElement('div');networkStructurePanel.className='panel';networkStructurePanel.id='networkStructurePanel';networkStructurePanel.innerHTML='<div class="ph">Struttura della rete neurale · pesi correnti</div><div class="pb"><canvas id="networkStructure" class="chart tall"></canvas><div class="small muted" id="networkStructureHint">La rete viene inizializzata quando avvii il training o esegui un passo.</div></div>';const lossSurfacePanel=document.createElement('div');lossSurfacePanel.className='panel';lossSurfacePanel.id='lossSurfacePanel';lossSurfacePanel.innerHTML='<div class="ph">Superficie loss · w₁ e w₂</div><div class="pb"><canvas id="nnSurface" class="chart tall"></canvas><div class="small muted">Sezione della loss attorno a due pesi della rete; si aggiorna dopo ogni passo.</div></div>';const gradientPanel=document.createElement('div');gradientPanel.className='panel';gradientPanel.id='gradientPanel';gradientPanel.innerHTML='<div class="ph">Pesi e bias · gradienti</div><div class="pb"><div class="tablewrap"><table class="data"><thead><tr><th>Parametro</th><th>Valore</th><th>∂L/∂param.</th></tr></thead><tbody id="nnParams"></tbody></table></div></div>';const backpropPanel=document.createElement('div');backpropPanel.className='panel';backpropPanel.id='backpropPanel';backpropPanel.innerHTML='<div class="ph">Spiegazione backpropagation</div><div class="pb"><div id="nnExplain" class="note">Avvia il training o esegui un passo per vedere come la backpropagation aggiorna i parametri.</div></div>';lossPanel.id='lossPanel';learningGrid.append(lossPanel,weightsPanel,hist,networkStructurePanel,lossSurfacePanel,gradientPanel,backpropPanel);learningGrid.prepend(networkStructurePanel);
datasetPanel.dataset.area='dataset';config.dataset.area='model';lossPanel.dataset.area='train';weightsPanel.dataset.area='model';dataChart.dataset.area='data visual';hist.dataset.area='history';sample.dataset.area='data';prediction.dataset.area='model';inputs.dataset.area='model';
datasetPanel.querySelectorAll('.note')[1].textContent='Tutti gli algoritmi utilizzano le feature selezionate. I grafici spaziali mostrano le prime due; la proiezione usa anche la terza feature quando disponibile.';
const tutorialData={
 logistic:['Classifica due classi stimando una probabilità.','Un filtro antispam assegna una probabilità che un messaggio sia spam.',['Legge le feature del campione, per esempio parole e mittente.','Combina gli input con pesi e bias e applica la sigmoide per ottenere una probabilità.','Confronta la probabilità con la soglia e misura l’errore rispetto alla classe nota.']],
 softmax:['Estende la classificazione probabilistica a più classi.','Un’immagine può essere classificata come gatto, cane o uccello: le probabilità sommano a 1.',['Calcola un punteggio per ogni classe.','La softmax converte i punteggi in probabilità confrontabili.','La cross-entropy valuta le probabilità e il gradiente aggiorna i pesi di tutte le classi.']],
 perceptron:['Impara un confine lineare aggiornandosi sugli esempi classificati male.','Un sensore decide se un oggetto è sopra o sotto una soglia, spostando la retta quando sbaglia.',['Calcola il punteggio lineare w·x+b.','Il segno del punteggio sceglie una delle due classi.','Se sbaglia, sposta pesi e bias nella direzione che corregge la previsione.']],
 linear:['Prevede un valore numerico combinando le feature.','Stima il prezzo di una casa a partire da superficie e numero di stanze.',['Legge le feature e applica la trasformazione φ scelta.','Calcola la previsione sommando i contributi pesati e l’intercetta.','Misura l’errore quadratico medio e modifica i coefficienti per ridurlo.']],
 ridge:['È una regressione che limita coefficienti troppo grandi con una penalità L2.','Con feature molto simili, Ridge distribuisce il loro contributo invece di far esplodere i pesi.',['Costruisce la previsione come una regressione lineare o polinomiale.','Calcola l’errore e aggiunge una penalità proporzionale ai quadrati dei pesi.','Aggiorna i coefficienti bilanciando accuratezza e semplicità del modello.']],
 lasso:['Regolarizza la regressione e può azzerare coefficienti, selezionando feature.','Per stimare un prezzo da molte misure, può eliminare quelle poco utili.',['Calcola la previsione usando tutte le feature trasformate.','Alla loss aggiunge la somma dei valori assoluti dei coefficienti.','Un aggiornamento prossimale riduce o azzera i pesi meno utili.']],
 elastic:['Combina la selezione delle feature L1 e la regolarizzazione L2.','Quando le misure sono correlate, Elastic Net seleziona un gruppo di segnali e riduce i pesi.',['Calcola una previsione lineare o polinomiale.','La loss combina errore, penalità L1 e penalità L2 secondo ρ.','Aggiorna i coefficienti applicando entrambe le penalità.']],
 knn:['Classifica un punto in base alle etichette dei k esempi più vicini; non costruisce una frontiera durante il training.','Per riconoscere un frutto, confronta colore e peso con frutti già etichettati.',['Scegli un campione da prevedere e calcola la sua distanza dai dati memorizzati.','Ordina le distanze e seleziona i k vicini più prossimi.','Assegna la classe più votata (eventualmente pesando i voti per distanza).']],
 svmrbf:['Usa il kernel RBF per trovare un confine non lineare con margine ampio.','Separa dati disposti in cerchi confrontando le somiglianze locali tra esempi.',['Confronta i campioni tramite il kernel RBF, controllato da γ.','L’ottimizzazione trova i support vector che definiscono il margine.','Il parametro C bilancia margine ampio e penalità per gli errori.']],
 svmpoly:['Usa un kernel polinomiale per modellare interazioni e confini curvi.','Una caratteristica come x₁·x₂ può distinguere configurazioni che una retta non separa.',['Calcola similarità polinomiali tra esempi usando grado, γ e coef0.','Trova i support vector e il confine con il margine più ampio.','C controlla quanto penalizzare i campioni che violano il margine.']],
 tree:['Divide i dati con domande sulle feature e conclude con una classe nella foglia.','Per approvare un prestito, controlla prima il reddito, poi eventuali altri criteri.',['Nel nodo corrente prova feature e soglie candidate.','Sceglie lo split che riduce maggiormente Gini o entropia.','Ripete le domande fino alla foglia, che restituisce la classe prevalente.']],
 forest:['Combina molti alberi addestrati su campioni e feature casuali.','Un gruppo di alberi vota se una transazione è sospetta; la maggioranza riduce la dipendenza da un singolo albero.',['Crea diversi campioni bootstrap e sottoinsiemi di feature.','Ogni albero impara una sequenza di split indipendente.','Somma i voti e restituisce la classe più votata.']],
 boost:['Aggiunge alberi in sequenza: ognuno corregge gli errori rimasti.','Per stimare una domanda, ogni nuovo albero affina la previsione del precedente.',['Parte da una previsione iniziale semplice.','Misura gli errori correnti e addestra un albero a correggerli.','Aggiunge la correzione con intensità η e ripete per gli stadi successivi.']],
 naive:['Stima la classe più probabile usando prior e probabilità delle feature.','Un filtro antispam combina la frequenza delle parole con la probabilità iniziale di spam.',['Stima dai dati la frequenza iniziale di ogni classe.','Calcola quanto è probabile ogni feature dato ciascun gruppo.','Combina le probabilità e sceglie la classe più plausibile.']],
 mlp:['Una rete neurale compone strati di trasformazioni e non linearità.','Riconosce cifre combinando rilevatori di tratti semplici in forme più complesse.',['Ogni neurone combina input con pesi e bias, poi applica un’attivazione.','Lo strato finale produce una previsione; la loss confronta output e target.','La backpropagation porta l’errore all’indietro e aggiorna i pesi.']],
 sgd:['Aggiorna i parametri usando il gradiente di un campione o mini-batch.','Una bussola usa il segnale di errore corrente per scegliere un piccolo passo verso la loss più bassa.',['Seleziona il campione o mini-batch corrente.','Calcola il gradiente della loss rispetto ai parametri.','Sottrae α volte il gradiente e passa al campione successivo.']],
 momentum:['Accumula una velocità dagli aggiornamenti per stabilizzare e accelerare la discesa.','Una palla in discesa conserva parte della velocità e supera piccoli avvallamenti.',['Calcola il gradiente corrente.','Combina il gradiente con la velocità precedente usando μ.','Aggiorna i parametri seguendo la velocità accumulata.']],
 nesterov:['Come Momentum, ma valuta il gradiente nella posizione anticipata.','Prima di fare un passo, guarda dove porterebbe l’inerzia e corregge la direzione.',['Stima la posizione futura usando la velocità precedente.','Calcola il gradiente in quel punto anticipato.','Aggiorna velocità e parametri usando la correzione prospettica.']],
 adagrad:['Adatta il learning rate di ogni parametro accumulando i gradienti passati.','Una feature aggiornata spesso riceve passi progressivamente più piccoli.',['Calcola il gradiente del passo corrente.','Accumula il quadrato del gradiente per ogni parametro.','Normalizza ciascun aggiornamento per la sua storia e applica il passo.']],
 rmsprop:['Adatta il passo usando una media mobile dei gradienti al quadrato.','Dimentica gradualmente gradienti vecchi, così il passo resta sensibile ai cambiamenti recenti.',['Calcola il gradiente corrente.','Aggiorna la media mobile quadratica controllata da ρ.','Normalizza il gradiente con questa scala e aggiorna i parametri.']],
 adam:['Combina la media dei gradienti con una stima adattiva della loro scala.','Usa direzione e intensità recenti per scegliere aggiornamenti rapidi ma controllati.',['Calcola il gradiente del passo.','Aggiorna il primo momento (direzione) e il secondo momento (scala).','Corregge il bias iniziale e applica l’aggiornamento normalizzato.']],
 adamw:['Usa Adam e applica il weight decay separatamente dalla loss.','Riduce i pesi in modo controllato per limitare modelli troppo complessi.',['Calcola il gradiente e aggiorna i momenti come in Adam.','Applica l’aggiornamento adattivo ai parametri.','Riduce separatamente i pesi con λ, senza incorporare il decay nel gradiente della loss.']],
 kmeans:['Raggruppa i punti assegnandoli al centroide più vicino.','Organizza clienti in gruppi secondo le loro caratteristiche senza etichette predefinite.',['Inizializza k centroidi.','Assegna ciascun campione al centro più vicino.','Ricalcola i centroidi come media dei punti e ripeti.']],
 dbscan:['Trova gruppi di punti densi e identifica i punti isolati come rumore.','In una mappa, riconosce quartieri di punti compatti senza fissare prima il numero dei gruppi.',['Per ogni punto cerca i vicini entro ε.','I punti con almeno minPts vicini diventano nuclei che espandono i cluster.','Assegna i punti raggiungibili e lascia il rumore senza cluster.']],
 gmm:['Modella i dati come miscela probabilistica di distribuzioni gaussiane.','Descrive gruppi sovrapposti assegnando a ogni punto probabilità di appartenenza.',['Inizializza medie, covarianze e pesi delle componenti.','E-step: calcola la responsabilità di ogni componente per ciascun punto.','M-step: aggiorna i parametri e ripete finché le probabilità si stabilizzano.']]
};
let tutorialStep=0;
const tutorialSteps=()=>tutorialData[method.value]||['Descrive il metodo selezionato.','Valuta un esempio con le feature correnti.',['Esamina i dati in ingresso.','Calcola una previsione secondo il modello.','Valuta il risultato e aggiorna o applica il modello.']];
function renderTutorial(reset=true){
 if(reset){tutorialStep=0;$('tutorialRunResult').textContent='';}
 const [purpose,example,steps]=tutorialSteps(),id=method.value,option=method.selectedOptions[0],counter=$('tutorialStepCount');
 $('tutorialTitle').textContent=option?.textContent.trim()||id;
 $('tutorialPurpose').textContent=purpose;
 const dataset=$('dataset').selectedOptions[0]?.textContent.trim()||'dataset caricato';
 const datasetDetails=`Qui il dataset selezionato è “${dataset}”, con ${$('nSamples').textContent} campioni e ${$('nFeatures').textContent} feature`;
 $('tutorialExample').textContent=`${example} ${datasetDetails}${activeTask==='regression'?'.':` e ${$('nClasses').textContent} classi.`}`;
 counter.textContent=`PASSAGGIO ${tutorialStep+1} DI ${steps.length}`;
 $('tutorialStepTitle').textContent=['1 · Osserva i dati','2 · Costruisci la previsione','3 · Valuta e migliora'][tutorialStep];
 $('tutorialStepBody').textContent=steps[tutorialStep];
 $('tutorialPrevious').disabled=tutorialStep===0;
 $('tutorialNext').textContent=tutorialStep===steps.length-1?'Ricomincia ↺':'Avanti →';
 $('tutorialProgress').replaceChildren(...steps.map((_,index)=>{
  const dot=document.createElement('span');
  dot.className='tutorial-dot'+(index===tutorialStep?' active':index<tutorialStep?' complete':'');
  dot.setAttribute('aria-label',`Passaggio ${index+1}`);
  return dot;
 }));
 renderTutorialTrace();
}
function renderTutorialTrace(){
 const list=$('tutorialTraceSteps'),status=$('tutorialTraceStatus');
 if(!list||!status)return;
 const sourceLegend=$('methodSymbolLegend'),tutorialLegend=$('tutorialSymbolLegend');
 if(sourceLegend&&tutorialLegend)tutorialLegend.replaceChildren(...[...sourceLegend.children].map(item=>item.cloneNode(true)));
 renderNeuralTutorialVisualization();
 if(typeof window.getAlgorithmLearningTrace!=='function'){
  status.textContent='La traccia numerica sarà disponibile dopo l’inizializzazione del motore.';
  list.replaceChildren();
  return;
 }
 const trace=window.getAlgorithmLearningTrace();
 status.textContent=trace.status;
 renderTutorialVisualization();
 const formulaSources={
  model:$('methodModelFormula'),
  cost:$('methodCostFormula'),
  update:$('methodUpdateFormula')
 };
 const formulaLabels={model:'Formula del modello',cost:'Formula della loss / valutazione',update:'Formula dell’aggiornamento'};
 const algorithmId=method.value;
 const formulaKinds=(step,index)=>{
  const title=step.title.toLocaleLowerCase('it'),detail=step.detail.toLocaleLowerCase('it');
  const kinds=[];
  if(index===0||/feature|input|prodotto|prevision|kernel|distan|voto|punteggi|statistic|densit|split|foglia|probabilit|modello|strato|centroide|regione|assegnazion/.test(title))kinds.push('model');
  if(/loss|errore|gradiente|margine|obiettivo|residu|cross.?entropy|log.?probabilit|inertia|valutazion|impurit/.test(title)||/J del campione|hinge loss/.test(detail))kinds.push('cost');
  if(/aggiornament|backpropagation|ottimizzator|m.step|centroide|gradient descent/.test(title)||/aggiornamento reale|θ₀ nuovo|Δθ/.test(detail))kinds.push('update');
  if(algorithmId==='knn'&&/voto/.test(title))kinds.push('cost','update');
  if(['tree','forest'].includes(algorithmId)&&/split/.test(title))kinds.push('cost');
  if(!kinds.length)kinds.push(index===trace.steps.length-1?'update':'model');
  return [...new Set(kinds)];
 };
 list.replaceChildren(...trace.steps.map((step,index)=>{
  const item=document.createElement('li'),title=document.createElement('strong'),detail=document.createElement('pre');
  title.textContent=step.title;
  detail.textContent=step.detail;
  item.append(title,detail);
  const formulas=document.createElement('div');
  formulas.className='tutorial-trace-formulas';
  for(const kind of formulaKinds(step,index)){
   const source=formulaSources[kind];
   if(!source)continue;
   const card=document.createElement('div'),label=document.createElement('span'),formula=source.cloneNode(true);
   card.className='tutorial-trace-formula';
   label.className='tutorial-trace-formula-label';
   label.textContent=formulaLabels[kind];
   formula.removeAttribute('id');
   formula.classList.add('tutorial-trace-formula-value');
   card.append(label,formula);
   formulas.append(card);
  }
  if(formulas.childElementCount)item.append(formulas);
  if(step.fullDetail){
   const disclosure=document.createElement('details'),summary=document.createElement('summary'),full=document.createElement('pre');
   summary.textContent='Mostra tutti i valori e i contributi';
   full.textContent=step.fullDetail;
   disclosure.append(summary,full);
   item.append(disclosure);
  }
  return item;
 }));
}
function clearFormulaHighlights(){
 document.querySelectorAll('.formula-component-active,.trace-related-line,.trace-related-value').forEach(element=>element.classList.remove('formula-component-active','trace-related-line','trace-related-value'));
}
function highlightFormulaValues(component){
 clearFormulaHighlights();
 const token=component.dataset.formulaToken||'',key=token.replace(/[^\p{L}\p{N}]/gu,'').toLocaleLowerCase('it');
 if(!key)return;
 document.querySelectorAll('.formula-component').forEach(element=>{if(element.dataset.formulaToken===token)element.classList.add('formula-component-active');});
 const candidates=[key,...(component.textContent.replace(/⁽[^⁾]*⁾/g,' ').replace(/[ᵀ₀-₉⁰-⁹]/g,' ').match(/[\p{L}\p{M}\p{N}]+/gu)||[]).map(part=>part.normalize('NFKC').replace(/[^\p{L}\p{N}]/gu,'').toLocaleLowerCase('it'))];
 for(const candidate of [...candidates])if(candidate.length>1)candidates.push(candidate.slice(0,-1));
 const escaped=candidates.filter(Boolean).sort((a,b)=>b.length-a.length).map(value=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'));
 if(!escaped.length)return;
 const matcher=new RegExp(`(^|[^\\p{L}\\p{N}])(?:${escaped.join('|')})(?=$|[^\\p{L}\\p{N}])`,'iu');
 document.querySelectorAll('#tutorialTraceSteps pre').forEach(block=>{
  const lines=(block.textContent||'').split('\n'),fragment=document.createDocumentFragment();
  lines.forEach((line,index)=>{
   if(matcher.test(line)){
    const related=document.createElement('span');
    related.className='trace-related-line';
    const pieces=line.split(/(\[[^\]\n]*\]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)/gi);
    pieces.forEach(piece=>{
     if(!piece)return;
     if(/^\[[^\]\n]*\]$/.test(piece)||/^-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(piece)){
      const value=document.createElement('span');value.className='trace-related-value';value.textContent=piece;related.append(value);
     }else related.append(document.createTextNode(piece));
    });
    fragment.append(related);
   }else fragment.append(document.createTextNode(line));
   if(index<lines.length-1)fragment.append(document.createTextNode('\n'));
  });
  block.replaceChildren(fragment);
 });
}
const tutorialTraceList=$('tutorialTraceSteps');
tutorialTraceList?.addEventListener('pointerover',event=>{
 const component=event.target.closest('.formula-component');
 if(component&&!component.contains(event.relatedTarget))highlightFormulaValues(component);
});
tutorialTraceList?.addEventListener('pointerout',event=>{
 if(event.target.closest('.formula-component')&&!event.target.closest('.formula-component').contains(event.relatedTarget))clearFormulaHighlights();
});
tutorialTraceList?.addEventListener('focusin',event=>{
 const component=event.target.closest('.formula-component');
 if(component)highlightFormulaValues(component);
});
tutorialTraceList?.addEventListener('focusout',event=>{
 if(event.target.closest('.formula-component'))clearFormulaHighlights();
});
function renderNeuralTutorialVisualization(){
 const panel=$('tutorialNeuralNetwork'),canvas=$('tutorialNeuralNetworkChart');
 if(!panel||!canvas||typeof window.getNeuralNetworkLearningVisualization!=='function')return;
 const visualization=window.getNeuralNetworkLearningVisualization();
 panel.hidden=!visualization;
 if(!visualization)return;
 const rect=canvas.getBoundingClientRect();
 if(!rect.width)return;
 const dpr=window.devicePixelRatio||1,W=rect.width,H=rect.height||360,ctx=canvas.getContext('2d');
 canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);canvas.style.height=`${H}px`;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);ctx.fillStyle='#091a2a';ctx.fillRect(0,0,W,H);
 const maxHidden=20,hiddenCount=visualization.hiddenValues.length,hiddenIndices=Array.from({length:Math.min(hiddenCount,maxHidden)},(_,index)=>Math.round(index*(hiddenCount-1)/Math.max(1,Math.min(hiddenCount,maxHidden)-1))),inputCount=visualization.inputValues.length,outputCount=visualization.outputValues.length,padTop=52,padBottom=40,xs=[W*.14,W*.5,W*.86],positions=count=>Array.from({length:count},(_,index)=>padTop+(H-padTop-padBottom)*(index/Math.max(1,count-1))),inputY=positions(inputCount),hiddenY=positions(hiddenIndices.length),outputY=positions(outputCount),weightSets=[visualization.weightsInputHidden,visualization.weightsHiddenOutput],previousWeightSets=[visualization.previousWeightsInputHidden,visualization.previousWeightsHiddenOutput],maxWeight=Math.max(1e-8,...weightSets.flatMap(weights=>weights.flat().map(Math.abs)),...previousWeightSets.flatMap(weights=>weights?weights.flat().map(Math.abs):[])),linkCount=inputCount*hiddenIndices.length+hiddenIndices.length*outputCount,showWeightValues=linkCount<=28;
 $('tutorialNeuralNetworkNote').textContent=`Passo ${visualization.iteration} · campione [${visualization.input.slice(0,4).map(value=>value.toFixed(2)).join(', ')}${visualization.input.length>4?', …':''}]${hiddenCount>hiddenIndices.length?` · mostrati ${hiddenIndices.length} di ${hiddenCount} neuroni nascosti`:''}. ${visualization.iteration?'I pesi correnti sono confrontati con quelli del passo precedente. Esegui altri passi per osservare la variazione.':'Questi sono i pesi iniziali; esegui un passo reale per confrontare l’aggiornamento.'}`;
 const drawConnections=(inputWeights,outputWeights,previous=false)=>{
  const color=value=>value>=0?'#2995ff':'#ff9d37';
  const edge=(x1,y1,x2,y2,value,layer,row,column)=>{
   if(!Number.isFinite(value))return;
   const offset=previous?2:0,dx=x2-x1,dy=y2-y1,length=Math.hypot(dx,dy)||1,ox=-dy/length*offset,oy=dx/length*offset;
   ctx.strokeStyle=previous?'rgba(203,213,225,.55)':color(value);ctx.lineWidth=previous?Math.min(3,.5+Math.abs(value)/maxWeight*2.5):Math.min(5,.7+Math.abs(value)/maxWeight*4.3);ctx.globalAlpha=previous ? .75 : .9;ctx.setLineDash(previous?[4,4]:[]);ctx.beginPath();ctx.moveTo(x1+ox,y1+oy);ctx.lineTo(x2+ox,y2+oy);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
   if(!previous&&showWeightValues){const mx=(x1+x2)/2+ox,my=(y1+y2)/2+oy+((row+column)%2?8:-7),label=`${value>=0?'+':''}${value.toFixed(2)}`;ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';const width=ctx.measureText(label).width+6;ctx.fillStyle='#091a2a';ctx.fillRect(mx-width/2,my-8,width,12);ctx.fillStyle='#dceaff';ctx.fillText(label,mx,my+1);}
  };
  for(let visible=0;visible<hiddenIndices.length;visible++){const hidden=hiddenIndices[visible];for(let input=0;input<inputCount;input++)edge(xs[0]+10,inputY[input],xs[1]-10,hiddenY[visible],inputWeights[hidden][input],1,hidden,input);for(let output=0;output<outputCount;output++)edge(xs[1]+10,hiddenY[visible],xs[2]-10,outputY[output],outputWeights[output][hidden],2,output,hidden);}
 };
 if(visualization.previousWeightsInputHidden&&visualization.previousWeightsHiddenOutput)drawConnections(previousWeightSets[0],previousWeightSets[1],true);
 drawConnections(weightSets[0],weightSets[1],false);
 const subscript=value=>String(value).replace(/\d/g,d=>'₀₁₂₃₄₅₆₇₈₉'[Number(d)]),nodeRadius=Math.min(15,Math.max(7,H/Math.max(inputCount,hiddenIndices.length,outputCount)/2.6));
 const drawNodes=(x,ys,values,labels,color)=>{
  values.forEach((value,index)=>{const y=ys[index];ctx.beginPath();ctx.fillStyle=color;ctx.arc(x,y,nodeRadius,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#dceaff';ctx.lineWidth=1;ctx.stroke();ctx.fillStyle='#fff';ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(Number(value||0).toFixed(2),x,y+3);ctx.fillStyle='#a9c9ef';ctx.fillText(labels(index),x,y+nodeRadius+15);});
 };
 ctx.fillStyle='#9db6d4';ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText('Input xᵢ',xs[0],22);ctx.fillText('Nascosto a⁽¹⁾',xs[1],22);ctx.fillText('Output ŷₖ',xs[2],22);ctx.fillStyle='#dceaff';ctx.fillText('W⁽¹⁾',(xs[0]+xs[1])/2,40);ctx.fillText('W⁽²⁾',(xs[1]+xs[2])/2,40);
 drawNodes(xs[0],inputY,visualization.inputValues,index=>`x${subscript(index+1)}`,'#236cb3');
 drawNodes(xs[1],hiddenY,hiddenIndices.map(index=>visualization.hiddenValues[index]),index=>`a⁽¹⁾${subscript(hiddenIndices[index]+1)}`,'#7650b5');
 drawNodes(xs[2],outputY,visualization.outputValues,index=>`ŷ${subscript(index+1)}`,'#16875e');
}
function renderTutorialVisualization(){
 const panel=$('tutorialVisualization');
 if(!panel||typeof window.getAlgorithmLearningVisualization!=='function')return;
 const selector=$('tutorialSampleIndex'),requestedIndex=Math.max(0,Number(selector.value||1)-1),sampleIndex=tutorialHoveredSampleIndex??requestedIndex,visualization=window.getAlgorithmLearningVisualization(sampleIndex),canvas=$('tutorialVisualizationChart'),note=$('tutorialVisualizationNote'),head=$('tutorialDistanceHead'),body=$('tutorialDistanceRows'),detailsTitle=$('tutorialAlgorithmDetailTitle'),detailsBody=$('tutorialAlgorithmDetails');
 panel.hidden=false;
 const isGeneric=visualization.kind.startsWith('model-'),isRegression=visualization.kind==='regression',entries=isGeneric?visualization.points||[]:visualization.entries||[],fmt=value=>value===null||value===undefined||!Number.isFinite(value)?'—':typeof value==='number'?value.toFixed(4):String(value),changeLabel=value=>value===null||value===undefined?'—':`${value>1e-8?'+':''}${fmt(value)}`;
 selector.dataset.sampleCount=String(entries.length);
 if(tutorialHoveredSampleIndex===null&&(!Number.isInteger(Number(selector.value))||Number(selector.value)<1||Number(selector.value)>entries.length))selector.value=String(Math.min(entries.length||1,requestedIndex+1));
 tutorialSampleEntries=entries;
 if(sampleSuggestionsOpen&&tutorialHoveredSampleIndex===null)updateTutorialSampleSuggestions();
 const previewedSample=entries[Math.min(entries.length-1,sampleIndex)];
 $('tutorialSelectedSample').textContent=previewedSample?`${tutorialHoveredSampleIndex!==null?'Anteprima · ':''}Campione ${sampleIndex+1} di ${entries.length} · input [${previewedSample.x.slice(0,4).map(fmt).join(', ')}${previewedSample.x.length>4?', …':''}] · target ${fmt(previewedSample.y)} · previsione ${fmt(isGeneric?previewedSample.prediction:previewedSample.after)}.`:'Nessun campione nel training set.';
 $('tutorialVisualizationTitle').textContent=isGeneric?({ 'model-regression':'Mappa di previsione e residui','model-clustering':'Mappa dei cluster e dei prototipi','model-classification':'Regioni decisionali del modello'}[visualization.kind]||'Visualizzazione del modello'):isRegression?'Retta di regressione e residui':'Retta decisionale e distanza dalla classe corretta';
 note.textContent=isGeneric?`${visualization.message} Modello: ${visualization.algorithm} · passo ${visualization.iteration}. Cerca il numero del campione o fai clic su un punto o una riga per selezionarlo.`:isRegression?`Passo ${visualization.iteration||0}. I punti mostrano i target osservati; la curva verde è la previsione del modello e i segmenti verticali sono i residui (target − previsione). ${visualization.dimensions>1?'La curva varia x₁ mantenendo le altre feature alla loro media. ':''}${visualization.hasPrevious?'Viola tratteggiato = previsione prima dell’ultimo aggiornamento.':'Esegui un secondo passo per confrontare prima e dopo.'}`:`${visualization.message||''} Passo ${visualization.iteration||0}. Punti blu/arancioni = classi 0/1; segmenti perpendicolari = distanza proiettata (verde migliora, rosso peggiora). ${visualization.hasPrevious?'Viola tratteggiato = confine precedente; verde = confine attuale.':'Esegui un altro passo per confrontare i due confini.'}`;
 const headings=isGeneric?['Campione','x₁','x₂','Target','Risposta modello','Esito']:isRegression?['Campione','x₁','Target y','Previsione prima','Previsione dopo','Residuo prima','Residuo dopo','Variazione |residuo|']:['Campione','x₁','x₂','Classe reale','Distanza prima','Distanza dopo','Δ distanza corretta'];
 head.innerHTML=`<tr>${headings.map(label=>`<th>${label}</th>`).join('')}</tr>`;
 body.replaceChildren(...entries.map(entry=>{
  const cells=isGeneric?[entry.index+1,fmt(entry.x[0]),fmt(entry.x[1]),fmt(entry.y),fmt(entry.prediction),visualization.kind==='model-regression'?fmt(entry.y-entry.prediction):entry.y===entry.prediction?'corretto':'errore']:isRegression?[entry.index,fmt(entry.x[0]),fmt(entry.y),fmt(entry.before),fmt(entry.after),fmt(entry.beforeDistance),fmt(entry.afterDistance),changeLabel(entry.change)]:[entry.index,fmt(entry.x[0]),fmt(entry.x[1]),entry.y,fmt(entry.beforeDistance),fmt(entry.afterDistance),changeLabel(entry.change)];
  const row=document.createElement('tr');
  row.dataset.sampleIndex=String(isGeneric?entry.index:entry.index-1);row.className=entry.selected?'selected-sample':'';
  row.innerHTML=cells.map((value,index)=>`<td${isGeneric&&index===cells.length-1&&visualization.kind!=='model-regression'?` class="${entry.y===entry.prediction?'distance-improved':'distance-worsened'}"`:!isGeneric&&index===cells.length-1&&entry.change!==null?` class="${entry.change>=0?'distance-improved':'distance-worsened'}"`:''}>${value}</td>`).join('');
  return row;
 }));
 detailsTitle.textContent=visualization.details?.[0]?.title||'Dettaglio del campione selezionato';
 detailsBody.replaceChildren(...(visualization.details?.[0]?.rows||[]).map(item=>{const row=document.createElement('tr'),label=document.createElement('td'),value=document.createElement('td');label.textContent=item.label;value.textContent=fmt(item.value);row.append(label,value);return row;}));
 detailsBody.closest('.tutorial-algorithm-detail-wrap').hidden=!isGeneric||!visualization.details?.length;
 const rect=canvas.getBoundingClientRect();
 if(!rect.width)return;
 const dpr=window.devicePixelRatio||1,W=rect.width,H=330,ctx=canvas.getContext('2d');
 canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);canvas.style.height=`${H}px`;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);ctx.fillStyle='#091a2a';ctx.fillRect(0,0,W,H);
 const markSelectedSample=(x,y,index)=>{ctx.save();ctx.strokeStyle='#fff3b0';ctx.fillStyle='#ffd166';ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(x,y,12,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(x-17,y);ctx.lineTo(x-11,y);ctx.moveTo(x+11,y);ctx.lineTo(x+17,y);ctx.moveTo(x,y-17);ctx.lineTo(x,y-11);ctx.moveTo(x,y+11);ctx.lineTo(x,y+17);ctx.stroke();ctx.font='bold 11px system-ui';const label=`Campione ${index+1}`,labelWidth=ctx.measureText(label).width;ctx.fillStyle='#ffd166';ctx.fillRect(x+15,y-25,labelWidth+12,19);ctx.fillStyle='#172033';ctx.fillText(label,x+21,y-12);ctx.restore();};
 const pad={left:48,right:20,top:26,bottom:40},plotW=W-pad.left-pad.right,plotH=H-pad.top-pad.bottom;
 const finiteValues=isRegression?[...entries.flatMap(entry=>[entry.y,entry.before,entry.after].filter(Number.isFinite)),...(visualization.curve||[]).flatMap(point=>[point.before,point.after].filter(Number.isFinite))]:entries.flatMap(entry=>[entry.x[1],0].filter(Number.isFinite));
 if(!entries.length||!finiteValues.length){ctx.fillStyle='#9db6d4';ctx.font='13px system-ui';ctx.fillText('Nessun campione disponibile per il grafico.',pad.left,pad.top+20);return;}
 const min= Math.min(...finiteValues),max=Math.max(...finiteValues),margin=(max-min||1)*.12,lo=min-margin,hi=max+margin;
 const sx=value=>pad.left+value*plotW,sy=value=>pad.top+(hi-value)/(hi-lo)*plotH;
 ctx.strokeStyle='#29445f';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.left,pad.top);ctx.lineTo(pad.left,H-pad.bottom);ctx.lineTo(W-pad.right,H-pad.bottom);ctx.stroke();ctx.fillStyle='#a9c9ef';ctx.font='11px system-ui';
 if(isRegression){
  const curve=visualization.curve||[],xValues=entries.map(entry=>entry.x[0]||0),xMin=Math.min(...xValues),xMax=Math.max(...xValues),xFor=value=>pad.left+(value-xMin)/(xMax-xMin||1)*plotW;
  ctx.fillText('Target y / previsione',pad.left,15);ctx.fillText('Feature x₁',Math.max(pad.left,W/2-24),H-8);
  const drawCurve=(key,color,dash,width)=>{const points=curve.filter(point=>Number.isFinite(point[key]));if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();points.forEach((point,index)=>index?ctx.lineTo(xFor(point.x),sy(point[key])):ctx.moveTo(xFor(point.x),sy(point[key])));ctx.stroke();ctx.setLineDash([]);};
  if(visualization.hasPrevious)drawCurve('before','#c084fc',[5,4],2);
  drawCurve('after','#35d58a',[],2.5);
  entries.forEach(entry=>{const x=xFor(entry.x[0]||0),actualY=sy(entry.y),predictionY=sy(entry.after);ctx.strokeStyle=entry.afterDistance>=0?'#35d58a':'#ff657a';ctx.globalAlpha=.48;ctx.beginPath();ctx.moveTo(x,actualY);ctx.lineTo(x,predictionY);ctx.stroke();ctx.globalAlpha=1;ctx.fillStyle='#e7f2ff';ctx.beginPath();ctx.arc(x,actualY,entry.selected?4:2.7,0,Math.PI*2);ctx.fill();});
  const selectedIndex=entries.findIndex(entry=>entry.selected);if(selectedIndex>=0)markSelectedSample(xFor(entries[selectedIndex].x[0]||0),sy(entries[selectedIndex].y),selectedIndex);
  ctx.fillStyle='#e7f2ff';ctx.beginPath();ctx.arc(pad.left+6,25,3,0,Math.PI*2);ctx.fill();ctx.fillText('target osservati',pad.left+14,29);
  ctx.strokeStyle='#35d58a';ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(pad.left+116,25);ctx.lineTo(pad.left+132,25);ctx.stroke();ctx.fillStyle='#a9c9ef';ctx.fillText('previsione',pad.left+137,29);
  if(visualization.hasPrevious){ctx.strokeStyle='#c084fc';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pad.left+220,25);ctx.lineTo(pad.left+236,25);ctx.stroke();ctx.setLineDash([]);ctx.fillText('prima',pad.left+241,29);}
  tutorialPlotState={nearest:(x,y)=>{let nearest=-1,distance=Infinity;entries.forEach((entry,index)=>{const d=(xFor(entry.x[0]||0)-x)**2+(sy(entry.y)-y)**2;if(d<distance){distance=d;nearest=index;}});return nearest;}};
 }else if(isGeneric){
  const domain=visualization.domain,centerX=(domain.xMin+domain.xMax)/2,centerY=(domain.yMin+domain.yMax)/2,scale=Math.min(plotW/(domain.xMax-domain.xMin),plotH/(domain.yMax-domain.yMin)),px=value=>pad.left+plotW/2+(value-centerX)*scale,py=value=>pad.top+plotH/2-(value-centerY)*scale;
  const colorFor=value=>{if(visualization.kind==='model-regression'){const values=visualization.surface.flat(),lo=Math.min(...values),hi=Math.max(...values),t=(value-lo)/(hi-lo||1);return `hsl(${220-190*t} 72% ${24+24*t}%)`;}const hue=visualization.kind==='model-clustering'?((Number(value)+1)*71+190)%360:(Number(value)*67+202)%360;return `hsl(${hue} 58% 27%)`;};
  const cellW=plotW/visualization.surface.length,cellH=plotH/visualization.surface.length;
  for(let iy=0;iy<visualization.surface.length;iy++)for(let ix=0;ix<visualization.surface[iy].length;ix++){ctx.fillStyle=colorFor(visualization.surface[iy][ix]);ctx.fillRect(pad.left+ix*cellW,pad.top+plotH-(iy+1)*cellH,Math.ceil(cellW)+.5,Math.ceil(cellH)+.5);}
  ctx.strokeStyle='#6c89a5';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.left,pad.top);ctx.lineTo(pad.left,H-pad.bottom);ctx.lineTo(W-pad.right,H-pad.bottom);ctx.stroke();ctx.fillStyle='#a9c9ef';ctx.font='11px system-ui';ctx.fillText('x₂',pad.left,15);ctx.fillText('x₁',W-pad.right-12,H-8);
  const selectedPoint=visualization.points[visualization.selectedIndex],selectedX=px(selectedPoint.x[0]),selectedY=py(selectedPoint.x[1]);
  const supportSet=new Set(visualization.supportVectors.map(item=>item.index));
  if(visualization.algorithm==='knn'){const neighbors=visualization.details?.[0]?.rows||[];for(const neighbor of neighbors){const match=neighbor.label.match(/campione (\d+)/),index=match?Number(match[1])-1:-1,target=visualization.points[index];if(!target)continue;ctx.strokeStyle='#f8fafc';ctx.globalAlpha=.5;ctx.beginPath();ctx.moveTo(selectedX,selectedY);ctx.lineTo(px(target.x[0]),py(target.x[1]));ctx.stroke();ctx.globalAlpha=1;}}
  entries.forEach((entry,index)=>{const x=px(entry.x[0]),y=py(entry.x[1]),correct=entry.y===entry.prediction;ctx.fillStyle=visualization.kind==='model-regression'?'#f8fafc':entry.y===0?'#2995ff':entry.y<0?'#94a3b8':'#ff9d37';ctx.beginPath();ctx.arc(x,y,entry.selected?5:3.1,0,Math.PI*2);ctx.fill();if(visualization.kind!=='model-regression'&&!correct){ctx.strokeStyle='#ff657a';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.stroke();}if(supportSet.has(index)){ctx.strokeStyle='#facc15';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,7.5,0,Math.PI*2);ctx.stroke();}});
  const selectedMarkerPoint=visualization.points[visualization.selectedIndex];if(selectedMarkerPoint)markSelectedSample(px(selectedMarkerPoint.x[0]),py(selectedMarkerPoint.x[1]),visualization.selectedIndex);
  visualization.centroids.forEach(center=>{const x=px(center.x),y=py(center.y);ctx.fillStyle='#facc15';ctx.strokeStyle='#111827';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y-8);ctx.lineTo(x+8,y);ctx.lineTo(x,y+8);ctx.lineTo(x-8,y);ctx.closePath();ctx.fill();ctx.stroke();});
  tutorialPlotState={nearest:(x,y)=>{let nearest=-1,distance=Infinity;visualization.points.forEach((point,index)=>{const d=(px(point.x[0])-x)**2+(py(point.x[1])-y)**2;if(d<distance){distance=d;nearest=index;}});return nearest;}};
 }else{
  ctx.fillText('x₂',pad.left,15);ctx.fillText('x₁',W-pad.right-12,H-8);
  const xs=entries.map(entry=>entry.x[0]),ys=entries.map(entry=>entry.x[1]),xMin=Math.min(...xs),xMax=Math.max(...xs),yMin=Math.min(...ys),yMax=Math.max(...ys),xm=(xMax-xMin||1)*.08,ym=(yMax-yMin||1)*.08;
  const scale=Math.min(plotW/(xMax-xMin+2*xm),plotH/(yMax-yMin+2*ym)),centerX=(xMin+xMax)/2,centerY=(yMin+yMax)/2,px=value=>pad.left+plotW/2+(value-centerX)*scale,py=value=>pad.top+plotH/2-(value-centerY)*scale;
  const drawBoundary=(line,color,dash,width)=>{if(!line)return;const candidates=[];if(Math.abs(line.y)>1e-12){for(const x of [xMin-xm,xMax+xm]){const y=-(line.x*x+line.intercept)/line.y;if(y>=yMin-ym&&y<=yMax+ym)candidates.push([x,y]);}}if(Math.abs(line.x)>1e-12){for(const y of [yMin-ym,yMax+ym]){const x=-(line.y*y+line.intercept)/line.x;if(x>=xMin-xm&&x<=xMax+xm)candidates.push([x,y]);}}if(candidates.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(px(candidates[0][0]),py(candidates[0][1]));ctx.lineTo(px(candidates[1][0]),py(candidates[1][1]));ctx.stroke();ctx.setLineDash([]);};
  drawBoundary(visualization.boundary?.before,'#c084fc',[6,5],2);drawBoundary(visualization.boundary?.after,'#35d58a',[],2.5);
  entries.forEach(entry=>{const x=px(entry.x[0]),y=py(entry.x[1]);if(entry.afterDistance!==null&&visualization.boundary?.after){const {x:a,y:b}=visualization.boundary.after,norm=Math.hypot(a,b);if(norm>1e-12){const projectedDistance=(a*entry.x[0]+b*entry.x[1]+visualization.boundary.after.intercept)/norm,dx=a/norm*projectedDistance,dy=b/norm*projectedDistance;ctx.strokeStyle=entry.change===null?'#cbd5e1':entry.change>=0?'#35d58a':'#ff657a';ctx.globalAlpha=.35;ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(px(entry.x[0]-dx),py(entry.x[1]-dy));ctx.stroke();ctx.globalAlpha=1;}}ctx.fillStyle=entry.y===1?'#ff9d37':'#2995ff';ctx.beginPath();ctx.arc(x,y,entry.selected?5:3.2,0,Math.PI*2);ctx.fill();});
  const selectedIndex=entries.findIndex(entry=>entry.selected);if(selectedIndex>=0)markSelectedSample(px(entries[selectedIndex].x[0]),py(entries[selectedIndex].x[1]),selectedIndex);
 }
}
function addTutorialNavigation(){if(aside.querySelector('[data-nav="tutorial"]'))return;const button=document.createElement('button');button.type='button';button.dataset.nav='tutorial';button.textContent='▷  Guida didattica';aside.insertBefore(button,aside.querySelector('.asidehint'));}
addTutorialNavigation();
method.addEventListener('change',()=>renderTutorial(true));algo.addEventListener('change',()=>renderTutorial(true));$('dataset').addEventListener('change',()=>renderTutorial(false));$('classCount')?.addEventListener('change',()=>renderTutorial(false));
$('tutorialPrevious').addEventListener('click',()=>{tutorialStep=Math.max(0,tutorialStep-1);renderTutorial(false);});
$('tutorialNext').addEventListener('click',()=>{const steps=tutorialSteps()[2];tutorialStep=tutorialStep===steps.length-1?0:tutorialStep+1;renderTutorial(false);});
$('tutorialRunStep').addEventListener('click',()=>{$('step').click();$('tutorialRunResult').textContent=$('status').textContent;});
window.addEventListener('ml-playground-update',()=>{updateAlgorithmCapabilities();if(page==='tutorial')renderTutorialTrace();});
// The model selector in the configuration panel is the visible selector; the method selector drives formulas and parameters behind it.
algo.addEventListener('change',()=>{if(method.value!==algo.value){method.value=algo.value;method.dispatchEvent(new Event('change',{bubbles:true}))}updateView()});method.addEventListener('change',()=>{if(algo.value!==method.value)algo.value=method.value;updateView()});
const navs=[...aside.querySelectorAll('[data-nav]')];for(const old of navs){let b=old.cloneNode(true);old.replaceWith(b);b.addEventListener('click',()=>setPage(b.dataset.nav))}function supportsNN(){return ['mlp','sgd','momentum','nesterov','adagrad','rmsprop','adam','adamw'].includes(method.value)}function supportsGradient(){return ['logistic','softmax','perceptron','linear','ridge','lasso','elastic','mlp','sgd','momentum','nesterov','adagrad','rmsprop','adam','adamw','svmrbf','svmpoly'].includes(method.value)}function supportsLossHistory(){return !['knn','tree','naive','dbscan'].includes(method.value)}
let page='home';function setPage(p){page=p;if(p==='tutorial'||p==='method'){tutorialControlBody.append(parameterSection);}else if(parameterSection){config.querySelector('.pb').insertBefore(parameterSection,$('split').previousElementSibling);}if(p==='method'){main.insertBefore(methodLab,grid);methodLab.querySelector('.pb').prepend(tutorialControls);}else if(p==='tutorial'){tutorial.querySelector('.pb').prepend(tutorialControls);tutorial.querySelector('.pb').append(methodLab);}aside.querySelectorAll('[data-nav]').forEach(b=>b.classList.toggle('active',b.dataset.nav===p));updateView();if(p==='tutorial')requestAnimationFrame(()=>renderTutorial());window.dispatchEvent(new Event('resize'));if(p==='train')requestAnimationFrame(()=>{const target=supportsNN()?networkStructurePanel:lossPanel;target.scrollIntoView({behavior:'smooth',block:'start'});})}function show(el,on){if(el)el.classList.toggle('nav-hidden',!on)}function updateView(){if(page==='home')datasetPanel.querySelector('.pb').append(dataChart);else rows.append(dataChart);const grad=supportsGradient(),histOK=supportsLossHistory();for(let el of [datasetPanel,config,lossPanel,weightsPanel,dataChart,hist,sample,prediction,inputs,results,methodLab,tutorial]){let areas=(el.dataset.area||'').split(' '),ok=page==='tutorial'?areas.includes('tutorial'):page==='method'?areas.includes('method'):page==='home'?!areas.includes('tutorial')&&!areas.includes('method')&&el!==tutorial&&el!==methodLab:(page==='data'&&areas.includes('dataset'))||(page==='train'&&(areas.includes('model')||areas.includes('train')||areas.includes('gradient')&&grad||areas.includes('history')&&histOK||areas.includes('results')))||(page==='visual'&&(areas.includes('data')||areas.includes('visual')||areas.includes('gradient')&&grad||areas.includes('history')&&histOK||areas.includes('results')));show(el,ok)}show(grid,[...grid.children].some(e=>!e.classList.contains('nav-hidden')));show(rows,[...rows.children].some(e=>!e.classList.contains('nav-hidden')));show(bottom,[...bottom.children].some(e=>!e.classList.contains('nav-hidden')));show(main.querySelector('.footer'),page==='home');aside.querySelectorAll('.asidehint, .small.muted, .note').forEach(e=>show(e,page==='home'));grid.dataset.page=page;bottom.dataset.page=page;}
const updatePageView=updateView;updateView=function(){updatePageView();const network=supportsNN(),restrict=(el,needed)=>{if(el)show(el,needed&&!el.classList.contains('nav-hidden'))};restrict($('#hyper'),false);restrict(inputs,false);restrict($('#nnConfigSettings'),network);restrict(sample,activeTask!=='clustering');restrict(weightsPanel,method.value!=='knn');restrict(lossPanel,true);show(learningData,['home','train','visual'].includes(page));show($('#accuracyChartPanel'),!lossPanel.classList.contains('nav-hidden'));for(const panel of [networkStructurePanel,lossSurfacePanel,gradientPanel,backpropPanel])restrict(panel,network);rows.style.gridTemplateColumns='minmax(0,1fr)';renderNetworkViews();}
function addChartZoomControls(canvas){
 if(canvas.dataset.zoomReady)return;
 const baseWidth=canvas.getBoundingClientRect().width,baseHeight=canvas.getBoundingClientRect().height,parent=canvas.parentElement;
 if(!parent||!baseWidth||!baseHeight)return;
 canvas.dataset.zoomReady='true';
 const viewport=document.createElement('div'),content=document.createElement('div'),controls=document.createElement('div'),state={zoom:1,width:baseWidth,height:baseHeight};
 viewport.className='chart-zoom-viewport';viewport.style.height=`${baseHeight}px`;
 content.className='chart-zoom-content';
 controls.className='chart-controls';
 controls.innerHTML='<button type="button" class="secondary" data-zoom="out" aria-label="Riduci lo zoom" title="Riduci lo zoom">−</button><button type="button" class="secondary" data-zoom="in" aria-label="Aumenta lo zoom" title="Aumenta lo zoom">+</button><button type="button" class="secondary" data-zoom="reset" aria-label="Ripristina la vista" title="Ripristina la vista">Ripristina vista</button><span class="small muted">Pinch / Ctrl+rotellina per zoom · due dita o trascina per spostarti</span>';
 canvas.before(viewport);viewport.append(content);content.append(canvas);parent.insertBefore(controls,viewport);
 state.width=viewport.clientWidth||baseWidth;
 const paddingX=()=>Math.max(64,state.width*.25),paddingY=()=>Math.max(56,state.height*.25);
 const resizeCanvas=()=>{const padX=paddingX(),padY=paddingY(),canvasWidth=state.width*state.zoom,canvasHeight=state.height*state.zoom;content.style.width=`${Math.max(viewport.clientWidth,canvasWidth)+padX*2}px`;content.style.height=`${Math.max(viewport.clientHeight,canvasHeight)+padY*2}px`;canvas.style.position='absolute';canvas.style.left=`${padX}px`;canvas.style.top=`${padY}px`;canvas.style.width=`${canvasWidth}px`;canvas.style.height=`${canvasHeight}px`;};
 const zoomAt=(factor,clientX,clientY)=>{const rect=viewport.getBoundingClientRect(),padX=paddingX(),padY=paddingY(),offsetX=clientX==null?viewport.clientWidth/2:clientX-rect.left,offsetY=clientY==null?viewport.clientHeight/2:clientY-rect.top,anchorX=(viewport.scrollLeft+offsetX-padX)/state.zoom,anchorY=(viewport.scrollTop+offsetY-padY)/state.zoom;state.zoom=Math.max(.5,Math.min(8,state.zoom*factor));resizeCanvas();viewport.scrollLeft=anchorX*state.zoom+paddingX()-offsetX;viewport.scrollTop=anchorY*state.zoom+paddingY()-offsetY;};
 controls.querySelector('[data-zoom="in"]').addEventListener('click',()=>zoomAt(1.25));
 controls.querySelector('[data-zoom="out"]').addEventListener('click',()=>zoomAt(.8));
 controls.querySelector('[data-zoom="reset"]').addEventListener('click',()=>{state.zoom=1;resizeCanvas();viewport.scrollLeft=paddingX();viewport.scrollTop=paddingY();});
 viewport.addEventListener('wheel',event=>{event.preventDefault();if(event.ctrlKey){zoomAt(Math.exp(-event.deltaY*.0015),event.clientX,event.clientY);return;}viewport.scrollLeft-=event.deltaX;viewport.scrollTop-=event.deltaY;},{passive:false});
 viewport.addEventListener('dblclick',()=>controls.querySelector('[data-zoom="reset"]').click());
 let pointer=null,touchGesture=null;
 const touchPoints=new Map();
 const distanceBetween=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
 const centerBetween=(a,b)=>({x:(a.x+b.x)/2,y:(a.y+b.y)/2});
 viewport.addEventListener('pointerdown',event=>{
  if(event.pointerType==='touch'){
   touchPoints.set(event.pointerId,{x:event.clientX,y:event.clientY});
   if(touchPoints.size===1)pointer={id:event.pointerId,x:event.clientX,y:event.clientY};
   viewport.setPointerCapture(event.pointerId);
   if(touchPoints.size===2){
    const [a,b]=touchPoints.values(),center=centerBetween(a,b);
    touchGesture={distance:Math.max(1,distanceBetween(a,b)),zoom:state.zoom,anchorX:(center.x-viewport.getBoundingClientRect().left+viewport.scrollLeft-paddingX())/state.zoom,anchorY:(center.y-viewport.getBoundingClientRect().top+viewport.scrollTop-paddingY())/state.zoom};
   }
   event.preventDefault();
   return;
  }
  if(event.button!==0)return;
  pointer={id:event.pointerId,x:event.clientX,y:event.clientY};
  viewport.setPointerCapture(event.pointerId);
  event.preventDefault();
 });
 viewport.addEventListener('pointermove',event=>{
  if(touchPoints.has(event.pointerId)){
   touchPoints.set(event.pointerId,{x:event.clientX,y:event.clientY});
   if(touchPoints.size>=2&&touchGesture){
    const [a,b]=touchPoints.values(),center=centerBetween(a,b),rect=viewport.getBoundingClientRect();
    state.zoom=Math.max(.5,Math.min(8,touchGesture.zoom*distanceBetween(a,b)/touchGesture.distance));
    resizeCanvas();
    viewport.scrollLeft=touchGesture.anchorX*state.zoom+paddingX()-(center.x-rect.left);
    viewport.scrollTop=touchGesture.anchorY*state.zoom+paddingY()-(center.y-rect.top);
   }else if(touchPoints.size===1){
    const point=touchPoints.get(event.pointerId),previous=pointer;
    if(previous?.id===event.pointerId){viewport.scrollLeft-=point.x-previous.x;viewport.scrollTop-=point.y-previous.y;}
    pointer={id:event.pointerId,x:point.x,y:point.y};
   }
   event.preventDefault();
   return;
  }
  if(!pointer||event.pointerId!==pointer.id)return;
  viewport.scrollLeft-=event.clientX-pointer.x;viewport.scrollTop-=event.clientY-pointer.y;pointer.x=event.clientX;pointer.y=event.clientY;
 });
 const release=event=>{
  if(touchPoints.has(event.pointerId)){touchPoints.delete(event.pointerId);touchGesture=null;pointer=null;}
  else if(pointer?.id===event.pointerId)pointer=null;
 };
 viewport.addEventListener('pointerup',release);viewport.addEventListener('pointercancel',release);viewport.addEventListener('lostpointercapture',release);
 window.addEventListener('resize',()=>{if(state.zoom===1){state.width=viewport.clientWidth||state.width;state.height=canvas.getBoundingClientRect().height||state.height;viewport.style.height=`${state.height}px`;resizeCanvas();viewport.scrollLeft=paddingX();viewport.scrollTop=paddingY();}});
 resizeCanvas();
 viewport.scrollLeft=paddingX();viewport.scrollTop=paddingY();
}
document.querySelectorAll('canvas.chart:not(#dataChart)').forEach(addChartZoomControls);
selectTask('classification');setPage('home');renderTutorial();
})();
