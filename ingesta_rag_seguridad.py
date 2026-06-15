import json
import requests

# Configuración de URLs de las APIs oficiales
CISA_KEV_URL = "https://cisa.gov"
EPSS_API_URL = "https://first.org"


def descargar_cisa_kev():
    """Descarga el catálogo de vulnerabilidades explotadas activamente de CISA."""
    print("[+] Descargando catálogo CISA KEV...")
    try:
        response = requests.get(CISA_KEV_URL, timeout=15)
        response.raise_for_status()
        return response.json().get("vulnerabilities", [])
    except Exception as e:
        print(f"[-] Error al descargar CISA KEV: {e}")
        return []


def obtener_datos_epss(lista_cves):
    """Consulta la API de FIRST EPSS para obtener scores en lote (máx 30 por request por eficiencia)."""
    print(f"[+] Consultando scores EPSS para {len(lista_cves)} CVEs...")
    epss_mapping = {}

    # Dividir las CVEs en bloques de 30 para no saturar la API pública
    chunk_size = 30
    for i in range(0, len(lista_cves), chunk_size):
        chunk = lista_cves[i : i + chunk_size]
        cves_param = ",".join(chunk)

        try:
            response = requests.get(
                EPSS_API_URL, params={"cve": cves_param}, timeout=10
            )
            if response.status_code == 200:
                data = response.json().get("data", [])
                for item in data:
                    epss_mapping[item["cve"]] = {
                        "epss": float(item["epss"]),
                        "percentile": float(item["percentile"]),
                    }
        except Exception as e:
            print(f"[-] Error consultando bloque EPSS: {e}")
            continue

    return epss_mapping


def estructurar_para_rag(vulnerabilidades, epss_data):
    """Cruza la información y genera la estructura de razonamiento para la Vector DB."""
    print("[+] Estructurando documentos con Cadena de Pensamiento (CoT)...")
    documentos_rag = []

    for vuln in vulnerabilities:
        cve_id = vuln.get("cveID")
        producto = vuln.get("product")
        vulnerabilidad_nombre = vuln.get("vulnerabilityName")
        descripcion = vuln.get("shortDescription")
        accion_requerida = vuln.get("requiredAction")

        # Obtener datos EPSS si existen, si no, usar valores por defecto
        epss_info = epss_data.get(cve_id, {"epss": 0.0, "percentile": 0.0})

        # --- AQUÍ ESTÁ EL TRUCO PARA EL RAG: CONSTRUIR EL TEXTO DE RAZONAMIENTO ---
        # Este bloque de texto será indexado como contenido del vector.
        # Cuando el LLM lo recupere, aprenderá a razonar el triage usando estos pasos.
        analisis_razonamiento = (
            f"Paso 1 (Identificación): Se analiza la vulnerabilidad {cve_id} que afecta a {producto} ({vulnerabilidad_nombre}).\n"
            f"Paso 2 (Validación de Amenaza Real): CISA confirma explotación activa en el mundo real. "
            f"EPSS reporta una probabilidad de explotación del {epss_info['epss']*100:.2f}% (Percentil {epss_info['percentile']*100:.2f}%).\n"
            f"Paso 3 (Análisis de Impacto sugerido): Si este componente está expuesto y activo en producción, "
            f"el nivel de urgencia debe elevarse inmediatamente a INMEDIATO debido a la explotación activa detectada.\n"
            f"Paso 4 (Acción de Mitigación requerida): {accion_requerida}."
        )

        # Documento JSON final listo para tu Pipeline de Incrustaciones (Embeddings)
        doc = {
            "id_cve": cve_id,
            "metadata": {
                "producto": producto,
                "cisa_kev": True,
                "epss_score": epss_info["epss"],
                "epss_percentile": epss_info["percentile"],
                "vulnerabilidad": vulnerabilidad_nombre,
            },
            "page_content": f"CVE: {cve_id}\nDescripción: {descripcion}\n\n[Análisis de Triage Automatizado]:\n{analisis_razonamiento}",
        }
        documentos_rag.append(doc)

    return documentos_rag


def main():
    # 1. Obtener vulnerabilidades críticas de CISA
    lista_cisa = descargar_cisa_kev()
    if not lista_cisa:
        print("[-] No se pudieron obtener datos. Cancelando proceso.")
        return

    # Extraer solo los IDs de las CVEs para consultar EPSS
    lista_cves = [v["cveID"] for v in lista_cisa if v.get("cveID")]

    # 2. Enriquecer con los scores de EPSS
    epss_data = obtener_datos_epss(lista_cves)

    # 3. Construir la base de conocimiento estructurada
    datos_finales_rag = estructurar_para_rag(lista_cisa, epss_data)

    # 4. Guardar en un archivo local
    archivo_salida = "base_conocimiento_triage.json"
    with open(archivo_salida, "w", encoding="utf-8") as f:
        json.dump(datos_finales_rag, f, indent=4, ensure_ascii=False)

    print(
        f"[\u2714] Proceso completado con éxito. {len(datos_finales_rag)} documentos guardados en '{archivo_salida}'."
    )


if __name__ == "__main__":
    main()

