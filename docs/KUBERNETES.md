# Kubernetes and GKE scenarios

How to run the agent harness against a Kubernetes cluster, with GKE as the
worked example. The harness works the same way against any cluster -- GKE is
chosen because it is the most common managed Kubernetes and has the richest
tooling surface for an agent to read.

The pattern is always the same: the harness runs as a sidecar (a container
alongside your application), reads cluster state through `kubectl`, and reports
or fixes what it finds. The deploy gate controls what it can change.

---

## Scenario 1: Monitoring a GKE workload

### What you get

An agent that runs on a cadence, reads your cluster's health, and reports:

- Pods that are crashlooping, pending, or not ready
- Deployments whose rollout is stuck
- Nodes under memory or CPU pressure
- Recent events that look like incidents
- Drift between what is deployed and what your manifests declare

### Setup

The harness runs as a Deployment in the same cluster it monitors. It needs:

1. A ServiceAccount with read-only RBAC
2. A ConfigMap with the provider catalogue and mission
3. A PersistentVolumeClaim for agent state
4. A Secret with the provider API key

### RBAC: read-only

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agent-harness
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: agent-harness-read
rules:
  # Pods, deployments, services, events, nodes -- read only.
  - apiGroups: [""]
    resources: ["pods", "pods/log", "services", "events", "nodes", "namespaces"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets", "statefulsets", "daemonsets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["batch"]
    resources: ["jobs", "cronjobs"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["metrics.k8s.io"]
    resources: ["nodes", "pods"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: agent-harness-read
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: agent-harness-read
subjects:
  - kind: ServiceAccount
    name: agent-harness
    namespace: monitoring
```

### ConfigMap: providers and mission

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: monitoring
data:
  providers.json: |
    {
      "default": {"provider": "example-cloud", "model": "vendor-model-flash"},
      "providers": {
        "example-cloud": {
          "base_url": "https://api.example-cloud.test/v1",
          "api_key_env": "EXAMPLE_CLOUD_API_KEY",
          "models": {"vendor-model-flash": "vendor-a/model-flash-1017"}
        }
      }
    }
  routes.json: |
    {
      "routes": [
        {"provider": "example-cloud", "model": "vendor-model-flash"}
      ]
    }
  schedule.json: |
    {
      "jobs": [{
        "id": "gke-health",
        "every_min": 30,
        "enabled": true,
        "message": "Check the health of all workloads in the cluster. Report any pods that are crashlooping, pending, or not ready. Check node resource pressure. Report recent warning events. Do not change anything."
      }]
    }
```

### Secret: provider key

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: agent-keys
  namespace: monitoring
type: Opaque
stringData:
  # The key for your provider. This is the only secret the harness needs.
  EXAMPLE_CLOUD_API_KEY: "your-key-here"
```

### Deployment: the harness sidecar

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-harness
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agent-harness
  template:
    metadata:
      labels:
        app: agent-harness
    spec:
      serviceAccountName: agent-harness
      containers:
        - name: agent
          image: agent-harness:1.0.0
          args: ["--serve", "--schedule"]
          env:
            - name: AGENT_PROJECT_ROOT
              value: /project
            - name: AGENT_PROVIDERS_FILE
              value: /agent-config/providers.json
            - name: AGENT_ROUTES_FILE
              value: /agent-config/routes.json
            - name: AGENT_SCHEDULE_FILE
              value: /agent-config/schedule.json
            - name: AGENT_SESSIONS_DIR
              value: /agent-data/sessions
            - name: AGENT_MEMORY_ROOT
              value: /agent-data/memories
            - name: AGENT_TRANSCRIPT
              value: /agent-data/transcript.jsonl
            - name: AGENT_MODEL_CHOICE_FILE
              value: /agent-data/sessions/model_choice.json
            - name: AGENT_RUN_SETTINGS_FILE
              value: /agent-data/sessions/run_settings.json
            - name: AGENT_SCHEDULE_STATE_FILE
              value: /agent-data/schedule_state.json
            - name: AGENT_SCHEDULE_HOLD
              value: /agent-data/paused.json
            - name: AGENT_WEBUI_HOST
              value: "0.0.0.0"
            - name: AGENT_WEBUI_PORT
              value: "8787"
            - name: AGENT_WEBUI_BASE_PATH
              value: "/agent"
            - name: AGENT_MAX_STEPS
              value: "200"
            - name: AGENT_MAX_RUN_SEC
              value: "3600"
            - name: AGENT_ROUTE_TIMEOUT_LIMIT
              value: "3"
            - name: AGENT_DEPLOY_MODE
              value: "disabled"  # read-only: no deploy gate needed
          envFrom:
            - secretRef:
                name: agent-keys
          volumeMounts:
            - name: config
              mountPath: /agent-config
              readOnly: true
            - name: data
              mountPath: /agent-data
          ports:
            - containerPort: 8787
          readinessProbe:
            httpGet:
              path: /agent/models
              port: 8787
            initialDelaySeconds: 10
            periodSeconds: 30
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
      volumes:
        - name: config
          configMap:
            name: agent-config
        - name: data
          persistentVolumeClaim:
            claimName: agent-data
```

### PersistentVolumeClaim

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agent-data
  namespace: monitoring
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 5Gi
```

### Accessing the console

The console has no built-in authentication. Do not expose it on a public IP.

```bash
# Port-forward for local access
kubectl port-forward -n monitoring svc/agent-harness 8787:8787
# Then open http://127.0.0.1:8787/agent
```

For team access, put an authenticated proxy in front of it:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-console
  namespace: monitoring
  annotations:
    # Use your cluster's auth proxy (e.g. oauth2-proxy, IAP, etc.)
    kubernetes.io/ingress.class: nginx
spec:
  rules:
    - host: agent.internal.example.com
      http:
        paths:
          - path: /agent
            pathType: Prefix
            backend:
              service:
                name: agent-harness
                port:
                  number: 8787
```

---

## Scenario 2: Gated remediation

When you want the agent to *fix* things, not just report them, you add the
deploy gate. The agent stages remediation requests; a separate applier pod
(or a host process) drains them.

### The pattern

```
Agent pod                    Applier pod (or host process)
    │                              │
    │  1. Stage request            │
    │  ────────────────────────>   │  (shared PVC or host path)
    │                              │
    │                              │  2. Read request
    │                              │  3. Check policy (RBAC, readiness)
    │                              │  4. Apply or refuse
    │                              │  5. Write receipt
    │                              │
    │  6. Read receipt             │
    │  <────────────────────────   │
```

### What the agent can remediate

With the gate in place, the agent can request:

| Remediation | How | Gate check |
|---|---|---|
| Restart a crashlooping pod | `kubectl delete pod <name>` | Pod is in CrashLoopBackOff |
| Scale a deployment | `kubectl scale deploy <name> --replicas=N` | Current replicas < desired |
| Rollback a bad deploy | `kubectl rollout undo deploy/<name>` | Rollout is stuck |
| Clear a stuck job | `kubectl delete job <name>` | Job is past its deadline |

The policy declares which actions are allowed, which fields they accept, and
what readiness state the cluster must be in before any of them runs.

### RBAC for remediation

Add write permissions to a SEPARATE ServiceAccount used by the applier, not
the agent. The agent stays read-only; the applier is the only thing that can
write.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: agent-applier
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["delete"]  # restart a pod
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["patch"]   # scale or rollback
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["delete"]  # clear a stuck job
```

---

## Scenario 3: Multi-cluster monitoring

Run one harness per cluster, each with its own ServiceAccount and ConfigMap.
A central dashboard (or just the console's run history) aggregates the reports.

```
Cluster A                    Cluster B                    Cluster C
  agent-harness                agent-harness                agent-harness
    │                            │                            │
    └────────────────────────────┴────────────────────────────┘
                                 │
                          Central report
                          (read sessions from each)
```

Each harness writes its sessions to a shared GCS bucket (via a sidecar or a
cron job), and a reader aggregates them. The harness itself does not need to
know about other clusters.

---

## What to put in your mission

For a GKE monitoring mission, the prompt should be specific about:

1. **Which namespaces** to check (not "all of them" -- that is too vague)
2. **What counts as a finding** (a crashloop, a stuck rollout, a node under
   pressure, a warning event)
3. **What NOT to do** (do not delete pods, do not scale deployments, do not
   change network policies)
4. **What evidence to cite** (the pod name, the event timestamp, the metric
   value)

Example:

```
Check all namespaces for pods that are not in Running or Completed state.
For each one, read its events and its last 50 log lines. Report the pod name,
namespace, status, and the most likely cause. Check node resource pressure
with `kubectl top nodes`. Report any node above 80% memory or CPU. Do not
change anything. Cite the exact command and output for every finding.
```

---

## Cost and resource notes

| Resource | Typical usage | Notes |
|---|---|---|
| CPU | 100m idle, 500m during a pass | The agent is mostly waiting on the model |
| Memory | 256Mi idle, 512Mi during a pass | `run_python` loads data into memory |
| Storage | 5Gi PVC | Sessions, memories, transcripts |
| Network | Egress to the provider endpoint | No ingress needed for monitoring |
| API calls | 1 per turn, ~20-60 turns per pass | Depends on the mission's scope |

The expensive part is the model, not the compute. A 30-minute cadence with a
30-turn pass costs ~30 model calls per hour. Choose your provider accordingly.
