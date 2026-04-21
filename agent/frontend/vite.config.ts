import { defineConfig } from "vite";

export default defineConfig({
  // Frontend is served directly by its own nginx container.
  base: "/",
});
