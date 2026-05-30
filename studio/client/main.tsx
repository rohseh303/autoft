import React from "react";
import { createRoot } from "react-dom/client";
import "./design/tokens.css";
import "./design/app.css";
import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
