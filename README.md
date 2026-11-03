# Machine Learning Playground

A local React application for exploring classification, regression, clustering, and neural-network optimizers. Training and CSV processing run entirely in the browser.

Requires Node.js 20.19+ or 22.12+ and npm.

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. Use the task tabs to select a problem, configure its dataset and algorithm, then train continuously or take one step. Pausing retains the current model; changing experiment settings starts a fresh model. CSV files accept comma, semicolon, or tab delimiters, an optional header, at least two numerical features, and a numerical target in the last column.

```sh
npm test                 # Numerical engine, model families, and CSV regression checks
npm run test:browser     # Browser workflows, canvas rendering, and mobile navigation
npm run build           # Production files in dist/
npm run preview         # Serve the production build locally
npm run format          # Format source, tests, and configuration
```

Browser tests use an installed Google Chrome. Edit `channel` in `playwright.config.js` to use another supported browser.

```text
src/
  App.jsx                       Page composition and navigation
  main.jsx                      React entry point
  components/                   Dataset, model, training, chart, tutorial, and result views
  hooks/usePlayground.js         Experiment state, CSV loading, timer lifecycle, pause/resume
  data/
    catalog.js                  Compatible tasks, algorithms, dataset names
    datasets.js                 Synthetic dataset generators
    csv.js                      Numeric CSV validation and target mapping
    methods.js                  Formulas, symbol glossary, parameter schemas
  engine/
    createEngine.js             Isolated model state, fitting, prediction, evaluation
    math.js                     Shared numerical and statistical helpers
    neural.js                   Forward pass, backpropagation, optimizer updates
    trees.js                    Classification tree construction
    learningInspection.js       Numerical tutorial traces and visualization descriptions
    charts.js                   Canvas drawing functions, with explicit data inputs
  styles/                       Shared appearance and responsive component layouts
tests/
  engine.test.js                Numerical and import regression coverage
  browser/playground.spec.js    End-to-end user workflow
```

React owns the UI and settings. `usePlayground` creates an isolated engine per experiment and cancels its training timer when settings change or the component unmounts. The engine has no DOM or React dependency, so algorithms can be tested directly in Node. Canvas components draw through effects and release their observers and listeners on unmount.

Code comments document the mathematical updates, numerical safeguards, model boundaries, and lifecycle decisions. Hyperparameter entries in `data/methods.js` are tuples of `[key, label, defaultValue, inputType]`; the model panel renders those schemas directly.

The previous standalone HTML and imperative UI scripts have been replaced. Start the app through Vite; `index.html` is the single application entry point.
