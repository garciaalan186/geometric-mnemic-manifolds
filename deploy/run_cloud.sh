#!/bin/bash
set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
CLUSTER_NAME="gmm-benchmark-cluster"
IMAGE_NAME="gcr.io/$PROJECT_ID/gmm-benchmark:latest"

echo "================================================================================"
echo "GMM CLOUD BENCHMARK KIT"
echo "Project: $PROJECT_ID"
echo "================================================================================"

# 1. Enable APIs
echo "[1/5] Enabling Container Registry & GKE APIs..."
gcloud services enable container.googleapis.com containerregistry.googleapis.com

# 2. Build & Push Docker Image (Using Cloud Build to bypass local Docker requirement)
echo "[2/5] Building and Pushing Docker Image (Cloud Build)..."
cp deploy/Dockerfile .
gcloud builds submit --tag $IMAGE_NAME .
rm Dockerfile

# 3. Create GKE Cluster (Ephemeral)
echo "[3/5] Creating GKE Cluster (This may take 5-10 minutes)..."
# V2: 8 Nodes x 4 vCPUs = 32 vCPUs (Max Quota)
CLUSTER_NAME="gmm-benchmark-v2-cluster"
gcloud container clusters create $CLUSTER_NAME \
    --zone $REGION-a \
    --num-nodes 8 \
    --machine-type e2-standard-4 \
    --disk-size=50

# 4. Get Credentials
echo "[4/5] Configuring kubectl..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone $REGION-a

# 5. Run Benchmark Job
echo "[5/5] Submitting Distributed Benchmark Stack (V2)..."
# Substitute Project ID
sed "s/PROJECT_ID/$PROJECT_ID/g" deploy/k8s-v2.yaml | kubectl apply -f -

# 6. Stream Logs
echo "Waiting for pods to start (StatefulSet takes time)..."
sleep 30
echo "Streaming logs from V2 Benchmark Coordinator..."
kubectl wait --for=condition=ready pod -l job-name=gmm-benchmark-v2 --timeout=600s
kubectl logs -l job-name=gmm-benchmark-v2 -f > benchmark_v2.log
cat benchmark_v2.log

echo ""
echo "When finished, DELETE the cluster to avoid billing:"
echo "  gcloud container clusters delete $CLUSTER_NAME --zone $REGION-a"
echo "================================================================================"
