# Machine Learning Playground

A local React and TypeScript application for exploring classification, regression, clustering, and neural-network optimizers. Training and CSV processing run entirely in the browser.

Requires Node.js 20.19+ or 22.12+ and npm.

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. Use the task tabs to select a problem, configure its dataset and algorithm, then train continuously or take one step. Pausing retains the current model; changing experiment settings starts a fresh model. CSV files accept comma, semicolon, or tab delimiters, an optional header, at least two numerical features, and a numerical target in the last column.

```sh
npm run typecheck        # Strict TypeScript checks for source, configuration, and tests
npm test                 # Numerical engine, model families, and CSV regression checks
npm run test:browser     # Browser workflows, canvas rendering, and mobile navigation
npm run build           # Type-check, then generate production files in dist/
npm run preview         # Serve the production build locally
npm run format          # Format source, tests, and configuration
```

Browser tests use an installed Google Chrome. Edit `channel` in `playwright.config.ts` to use another supported browser.

```text
src/
  App.tsx                       Page composition and navigation
  main.tsx                      React entry point
  types.ts                      Shared configuration, model, snapshot, and chart contracts
  components/                   Dataset, model, training, chart, tutorial, and result views
  hooks/usePlayground.ts         Experiment state, CSV loading, timer lifecycle, pause/resume
  hooks/useSolver.ts             Worker lifecycle, commands, asynchronous predictions, frame commits
  data/
    catalog.ts                  Compatible tasks, algorithms, dataset names
    datasets.ts                 Synthetic dataset generators
    csv.ts                      Numeric CSV validation and target mapping
    methods.ts                  Formulas, symbol glossary, parameter schemas
  engine/
    createEngine.ts             Isolated model state, fitting, prediction, evaluation
    math.ts                     Shared numerical and statistical helpers
    neural.ts                   Forward pass, backpropagation, optimizer updates
    modelGuards.ts              Runtime model invariants and safe union narrowing
    modelParameters.ts          Bounded parameter table prepared inside the worker
    trees.ts                    Classification tree construction
    learningInspection.ts       Numerical tutorial traces and visualization descriptions
    charts.ts                   Canvas drawing functions, with explicit data inputs
    decisionSurfaces.ts         Clipped linear hyperplanes and nonlinear boundary meshes
  styles/                       Shared appearance and responsive component layouts
  solver/
    solver.worker.ts            Separate browser worker entry point
    runtime.ts                  Training slices, throttled display frames, pause/resume
    protocol.ts                 Typed messages and cached presentation data
tests/
  engine.test.ts                Numerical and import regression coverage
  solver.test.ts                Worker scheduling, frame backpressure, cancellation, restart
  browser/playground.spec.ts    End-to-end user workflow
  browser/responsiveness.spec.ts Large-fit navigation, chart zoom, animation, and reset coverage
  browser/decisionSurfaces.spec.ts 3D boundary geometry, rotation, and visibility controls
```

React owns the UI and settings. `usePlayground` delegates numerical work to `useSolver`, which creates a dedicated Web Worker per experiment. Settings changes, reset, and unmount terminate the old worker, including an expensive epoch already in progress. Pause preserves its model and optimizer state at an epoch boundary. The engine has no DOM or React dependency, so algorithms can still be tested directly in Node.

Training, prediction-grid sampling, tutorial calculations, and neural diagnostics run in the worker. Canvas components only project cached results and draw pixels, so dragging or zooming does not trigger synchronous inference. Display frames are capped at eight per second, with at most one frame awaiting a React commit; training continues independently without queuing obsolete renders. Models, forests, and training caches remain in the worker. Messages contain metrics, sampled surfaces, row predictions, and a bounded parameter table.

The 3D classification view shows learned decision boundaries. Linear classifiers use exact pairwise score-equality hyperplanes, clipped to the visible feature-space box and to regions where those classes beat all competitors. Nonlinear models use marching-tetrahedra boundary meshes. Features beyond the first three are held at their dataset means to form a cross-section of the original model. Boundary-pair controls hide individual surfaces; rotation and zoom reproject cached geometry without invoking the solver.

Live loss histories and network activations update during training. Exact neural loss surfaces and detailed gradient tables refresh on pause, single-step, or completion, since evaluating those surfaces requires hundreds of full-batch gradients. Their last computed iteration is identified while training. Canvas observers/listeners and worker resources are released on unmount.

The project uses strict TypeScript throughout, including the numerical engine, React components, Vite configuration, and Playwright tests. `npm run dev` serves the app; `npm run typecheck` checks types without emitting files, and production builds perform that check automatically.

`types.ts` defines a discriminated `Model` union: narrow `model.kind` before accessing family-specific arrays. A regression coefficient vector, a classifier weight matrix, and a neural network are deliberately different types. DOM and imported values are checked or normalized at their boundaries; the code does not use `any` annotations or TypeScript suppression directives.

Rich JSDoc and inline comments explain formulas, dimensions, invariants, numerical safeguards, data leakage prevention, timer cleanup, stale CSV cancellation, and chart coordinate systems. Start with `usePlayground`, then `createEngine`, and follow the model family into `math`, `neural`, or `trees`. Hyperparameter entries in `data/methods.ts` are tuples of `[key, label, defaultValue, inputType]`; the model panel renders those schemas directly.

The previous standalone HTML and imperative UI scripts have been replaced. Start the app through Vite; `index.html` is the single application entry point.
