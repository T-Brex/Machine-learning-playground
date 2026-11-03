import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./styles/playground.css";
import "./styles/react.css";

const container = document.getElementById("root");
if (!container) throw new Error("Missing React root container");
createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
