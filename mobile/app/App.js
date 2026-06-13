// CardioIA Fase 4 — App React Native (Expo) — Ir Além 2
// Fluxo: selecionar radiografia -> enviar ao backend Flask -> exibir classe,
// confiança e mapa de calor Grad-CAM. Protótipo educacional.
import React, { useState } from "react";
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import { BACKEND_URL } from "./config";

export default function App() {
  const [imagem, setImagem] = useState(null);
  const [carregando, setCarregando] = useState(false);
  const [resultado, setResultado] = useState(null);
  const [erro, setErro] = useState(null);

  async function escolherImagem() {
    setErro(null);
    setResultado(null);
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      setErro("Permissão de acesso às fotos negada.");
      return;
    }
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"], // API atual do expo-image-picker (SDK 54+)
      quality: 1,
    });
    if (!res.canceled) {
      setImagem(res.assets[0]);
    }
  }

  async function classificar() {
    if (!imagem) return;
    setCarregando(true);
    setErro(null);
    setResultado(null);
    try {
      const form = new FormData();
      form.append("image", {
        uri: imagem.uri,
        name: "raiox.png",
        type: "image/png",
      });
      const resp = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: form,
        headers: { "Content-Type": "multipart/form-data" },
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.erro || "Erro no servidor");
      setResultado(data);
    } catch (e) {
      setErro(`Falha ao classificar: ${e.message}. Verifique BACKEND_URL em config.js.`);
    } finally {
      setCarregando(false);
    }
  }

  const ehCardio = resultado?.classe === "cardiomegalia";

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <StatusBar style="light" />
      <Text style={styles.titulo}>🫀 CardioIA</Text>
      <Text style={styles.subtitulo}>Detecção de Cardiomegalia em raio-X de tórax</Text>

      <TouchableOpacity style={styles.botao} onPress={escolherImagem}>
        <Text style={styles.botaoTxt}>📁 Selecionar radiografia</Text>
      </TouchableOpacity>

      {imagem && (
        <Image source={{ uri: imagem.uri }} style={styles.preview} resizeMode="contain" />
      )}

      {imagem && (
        <TouchableOpacity
          style={[styles.botao, styles.botaoPrimario]}
          onPress={classificar}
          disabled={carregando}
        >
          <Text style={styles.botaoTxt}>{carregando ? "Analisando..." : "🔍 Classificar"}</Text>
        </TouchableOpacity>
      )}

      {carregando && <ActivityIndicator size="large" color="#4da3ff" style={{ marginTop: 16 }} />}

      {erro && <Text style={styles.erro}>{erro}</Text>}

      {resultado && (
        <View style={[styles.card, ehCardio ? styles.cardAlerta : styles.cardOk]}>
          <Text style={styles.resultadoLabel}>
            {ehCardio ? "🔴" : "🟢"} {resultado.label}
          </Text>
          <Text style={styles.confianca}>
            Confiança: {(resultado.confianca * 100).toFixed(1)}%
          </Text>
          <Text style={styles.prob}>
            P(cardiomegalia) = {(resultado.prob_cardiomegalia * 100).toFixed(1)}%
          </Text>
          {resultado.gradcam_png_b64 && (
            <>
              <Text style={styles.gradcamTitulo}>Grad-CAM (regiões analisadas)</Text>
              <Image
                source={{ uri: resultado.gradcam_png_b64 }}
                style={styles.gradcam}
                resizeMode="contain"
              />
            </>
          )}
          <Text style={styles.aviso}>{resultado.aviso}</Text>
        </View>
      )}

      <Text style={styles.rodape}>
        ⚠️ Protótipo educacional — não é dispositivo médico e não substitui avaliação clínica.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, backgroundColor: "#0b1f3a", padding: 20, alignItems: "center" },
  titulo: { fontSize: 34, fontWeight: "bold", color: "#fff", marginTop: 40 },
  subtitulo: { fontSize: 14, color: "#9bb8de", marginBottom: 24, textAlign: "center" },
  botao: {
    backgroundColor: "#1c3a63",
    paddingVertical: 14,
    paddingHorizontal: 22,
    borderRadius: 12,
    marginTop: 12,
    width: "100%",
    alignItems: "center",
  },
  botaoPrimario: { backgroundColor: "#2563eb" },
  botaoTxt: { color: "#fff", fontSize: 16, fontWeight: "600" },
  preview: { width: "100%", height: 280, marginTop: 18, borderRadius: 12, backgroundColor: "#000" },
  card: { width: "100%", borderRadius: 16, padding: 18, marginTop: 22 },
  cardOk: { backgroundColor: "#103a2a", borderColor: "#2ca02c", borderWidth: 1 },
  cardAlerta: { backgroundColor: "#3a1414", borderColor: "#d62728", borderWidth: 1 },
  resultadoLabel: { fontSize: 20, fontWeight: "bold", color: "#fff" },
  confianca: { fontSize: 16, color: "#e6eefc", marginTop: 8 },
  prob: { fontSize: 13, color: "#9bb8de", marginTop: 2 },
  gradcamTitulo: { color: "#9bb8de", marginTop: 16, marginBottom: 6, fontSize: 13 },
  gradcam: { width: "100%", height: 260, borderRadius: 10, backgroundColor: "#000" },
  aviso: { color: "#c9b66b", fontSize: 12, marginTop: 14, fontStyle: "italic" },
  erro: { color: "#ff7b7b", marginTop: 16, textAlign: "center" },
  rodape: { color: "#6f86a8", fontSize: 11, marginTop: 30, textAlign: "center", paddingBottom: 30 },
});
