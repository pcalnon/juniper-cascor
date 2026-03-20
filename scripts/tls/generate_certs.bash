#!/usr/bin/env bash
# ============================================================================
# generate_certs.bash — Generate TLS certificates for Juniper CasCor mTLS
#
# Creates a self-signed CA, server certificate, and worker client certificate
# for securing the WebSocket worker endpoint (/ws/v1/workers).
#
# Usage:
#   bash scripts/tls/generate_certs.bash [--output-dir DIR] [--days DAYS]
#
# Output:
#   <output-dir>/ca.crt         — CA certificate (distribute to all parties)
#   <output-dir>/ca.key         — CA private key (keep offline/secure)
#   <output-dir>/server.crt     — Server certificate (for juniper-cascor)
#   <output-dir>/server.key     — Server private key (for juniper-cascor)
#   <output-dir>/worker.crt     — Worker client certificate (for workers)
#   <output-dir>/worker.key     — Worker private key (for workers)
# ============================================================================

set -euo pipefail

OUTPUT_DIR="${HOME}/.juniper/tls"
DAYS=365
CA_SUBJECT="/C=US/O=Juniper/CN=Juniper CasCor CA"
SERVER_SUBJECT="/C=US/O=Juniper/CN=juniper-cascor"
WORKER_SUBJECT="/C=US/O=Juniper/CN=juniper-cascor-worker"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --days)
            DAYS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--output-dir DIR] [--days DAYS]"
            echo ""
            echo "Generate TLS certificates for Juniper CasCor mTLS."
            echo ""
            echo "Options:"
            echo "  --output-dir DIR   Output directory (default: ~/.juniper/tls)"
            echo "  --days DAYS        Certificate validity in days (default: 365)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Juniper CasCor TLS Certificate Generator"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Validity:         ${DAYS} days"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
chmod 700 "${OUTPUT_DIR}"

# --- CA Certificate ---
echo "[1/3] Generating CA certificate..."
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
    -keyout "${OUTPUT_DIR}/ca.key" \
    -out "${OUTPUT_DIR}/ca.crt" \
    -days "${DAYS}" \
    -nodes \
    -subj "${CA_SUBJECT}" \
    2>/dev/null
chmod 600 "${OUTPUT_DIR}/ca.key"
echo "       CA cert:  ${OUTPUT_DIR}/ca.crt"
echo "       CA key:   ${OUTPUT_DIR}/ca.key"

# --- Server Certificate ---
echo "[2/3] Generating server certificate..."
openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
    -keyout "${OUTPUT_DIR}/server.key" \
    -out "${OUTPUT_DIR}/server.csr" \
    -nodes \
    -subj "${SERVER_SUBJECT}" \
    2>/dev/null

# Server cert with SAN for localhost development
cat > "${OUTPUT_DIR}/server_ext.cnf" <<EXTEOF
[v3_ext]
subjectAltName = DNS:localhost,DNS:juniper-cascor,IP:127.0.0.1
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
EXTEOF

openssl x509 -req \
    -in "${OUTPUT_DIR}/server.csr" \
    -CA "${OUTPUT_DIR}/ca.crt" \
    -CAkey "${OUTPUT_DIR}/ca.key" \
    -CAcreateserial \
    -out "${OUTPUT_DIR}/server.crt" \
    -days "${DAYS}" \
    -extfile "${OUTPUT_DIR}/server_ext.cnf" \
    -extensions v3_ext \
    2>/dev/null
chmod 600 "${OUTPUT_DIR}/server.key"
rm -f "${OUTPUT_DIR}/server.csr" "${OUTPUT_DIR}/server_ext.cnf" "${OUTPUT_DIR}/ca.srl"
echo "       Server cert: ${OUTPUT_DIR}/server.crt"
echo "       Server key:  ${OUTPUT_DIR}/server.key"

# --- Worker Client Certificate ---
echo "[3/3] Generating worker client certificate..."
openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
    -keyout "${OUTPUT_DIR}/worker.key" \
    -out "${OUTPUT_DIR}/worker.csr" \
    -nodes \
    -subj "${WORKER_SUBJECT}" \
    2>/dev/null

cat > "${OUTPUT_DIR}/worker_ext.cnf" <<EXTEOF
[v3_ext]
keyUsage = digitalSignature
extendedKeyUsage = clientAuth
EXTEOF

openssl x509 -req \
    -in "${OUTPUT_DIR}/worker.csr" \
    -CA "${OUTPUT_DIR}/ca.crt" \
    -CAkey "${OUTPUT_DIR}/ca.key" \
    -CAcreateserial \
    -out "${OUTPUT_DIR}/worker.crt" \
    -days "${DAYS}" \
    -extfile "${OUTPUT_DIR}/worker_ext.cnf" \
    -extensions v3_ext \
    2>/dev/null
chmod 600 "${OUTPUT_DIR}/worker.key"
rm -f "${OUTPUT_DIR}/worker.csr" "${OUTPUT_DIR}/worker_ext.cnf" "${OUTPUT_DIR}/ca.srl"
echo "       Worker cert: ${OUTPUT_DIR}/worker.crt"
echo "       Worker key:  ${OUTPUT_DIR}/worker.key"

echo ""
echo "============================================================"
echo "Done. Files generated in: ${OUTPUT_DIR}"
echo ""
echo "To use with juniper-cascor server:"
echo "  export CASCOR_TLS_CERT=${OUTPUT_DIR}/server.crt"
echo "  export CASCOR_TLS_KEY=${OUTPUT_DIR}/server.key"
echo "  export CASCOR_TLS_CA=${OUTPUT_DIR}/ca.crt"
echo ""
echo "To use with juniper-cascor-worker:"
echo "  juniper-cascor-worker --tls-cert ${OUTPUT_DIR}/worker.crt \\"
echo "                        --tls-key ${OUTPUT_DIR}/worker.key \\"
echo "                        --tls-ca ${OUTPUT_DIR}/ca.crt"
echo "============================================================"
