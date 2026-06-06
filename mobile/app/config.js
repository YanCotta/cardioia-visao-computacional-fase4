// Configuração do app CardioIA mobile.
//
// IMPORTANTE: troque pelo IP da máquina que roda o backend Flask (porta 5050).
// - Emulador Android (AVD):      http://10.0.2.2:5050
// - iOS Simulator:               http://localhost:5050
// - Dispositivo físico (Expo Go): http://SEU_IP_LOCAL:5050  (ex.: 192.168.0.12)
//   descubra o IP com `ipconfig getifaddr en0` (macOS) e mantenha o celular na mesma rede.
export const BACKEND_URL = "http://192.168.0.12:5050";
