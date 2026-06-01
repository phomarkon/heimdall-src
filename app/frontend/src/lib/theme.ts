import type { PersonaArchetype, SocietyEdgeKind } from "@/types/heimdall";

export const archetypeColor: Record<PersonaArchetype, string> = {
  wind: "#2f8cc8",
  ev: "#7567d9",
  retailer: "#c455a3",
  p2h: "#17a99a",
  generator: "#c89520",
  arbitrageur: "#d45568",
  "grid-info": "#4f8f5b",
  "outage-info": "#8f6bb8",
  "price-info": "#5f87d8",
  "sizing-info": "#b8762b",
  "uncertainty-info": "#6b7280",
  "decision-info": "#9f7aea",
  "risk-info": "#c94054"
};

export const llmFamilyColor: Record<string, string> = {
  Qwen: "#0f9f8e",
  Gemma: "#2f8cc8",
  Mistral: "#b98112",
  Llama: "#7567d9",
  DeepSeek: "#c94054",
  Moonshot: "#c455a3"
};

export const edgeColor: Record<SocietyEdgeKind, string> = {
  consensus: "#17a99a",
  broadcast: "#5f87d8"
};
