import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import App from "./App";
import { applyFontSizes, loadFontSizes } from "./fontSizes";
import "./styles.css";

// Apply the persisted text sizes before the first paint so the viewer does not
// flash at the default size.
applyFontSizes(loadFontSizes());

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

const container = document.getElementById("root");
if (!container) {
  throw new Error("Missing #root element.");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
