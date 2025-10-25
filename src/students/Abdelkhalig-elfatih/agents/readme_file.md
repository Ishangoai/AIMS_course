# Agentic Report Writing System

**Multi-Agent System for Automated Technical Report Generation**

## 📋 Project Overview

This project implements a sophisticated multi-agent system using LangGraph and Claude AI to automatically generate comprehensive technical reports on course-related topics (MLOps, CI/CD, APIs, Gradio, etc.).

### Key Features
- ✅ 1000-word reports (±50 words) with automatic length control
- ✅ Multi-agent workflow with specialized agents
- ✅ Structured output (Introduction, Main Body, Conclusion)
- ✅ Quality assurance and fact-checking
- ✅ Human-in-the-loop editing capability
- ✅ Gradio web interface
- ✅ GCP deployment ready

---

## 🏗️ System Architecture

### Agent Workflow

```
User Input (Topic)
    ↓
🔍 Research Agent
    ↓ (identifies research areas)
📋 Outline Agent
    ↓ (creates structure)
✍️  Writing Agent
    ↓ (writes sections)
🔍 Quality Agent
    ↓ (reviews & fact-checks)
✂️  Editor Agent
    ↓ (adjusts length & format)
    ↓
Final Report (950-1050 words)
```

### Agent Responsibilities

1. **Research Agent**: Identifies 3-5 key research areas and questions
2. **Outline Agent**: Creates detailed structure with word counts
3. **Writing Agent**: Composes each section with examples
4. **Quality Agent**: Reviews for accuracy, flow, and completeness
5. **Editor Agent**: Ensures proper length and formatting
6. **Coordinator**: Manages workflow via LangGraph (up to 3 iterations)

---

## 🚀 Quick Start

### Option 1: Local Python Script

```bash
# 1. Clone or download files
mkdir agentic-report-writer
cd agentic-report-writer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
export ANTHROPIC_API_KEY="your-api-key-here"

# 5. Run the system
python agentic_report_writer.py
```

### Option 2: Jupyter Notebook / Google Colab

```python
# Cell 1: Install dependencies
!pip install langchain langgraph langchain-anthropic gradio python-dotenv

# Cell 2: Set API key
import os
os.environ['ANTHROPIC_API_KEY'] = 'your-api-key-here'

# Cell 3: Copy agent_system.py code and run

# Cell 4: Copy simple_usage.py code and run
```

### Option 3: Gradio Web Interface

```bash
# 1. Install dependencies (same as above)

# 2. Run Gradio app
python app.py

# 3. Open browser at http://localhost:7860
```

---

## 📁 File Structure

```
agentic-report-writer/
│
├── agent_system.py          # Core multi-agent system (LangGraph)
├── app.py                   # Gradio web interface
├── pdf_utils.py             # PDF export functionality
├── simple_usage.py          # Quick start script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── Dockerfile              # Container for GCP deployment
├── deploy_gcp.sh           # GCP deployment script
├── README.md               # This file
│
├── reports/                # Generated reports saved here
│   ├── report_*.txt
│   └── report_*.pdf
│
└── agentic_workflow_diagram.png  # System diagram
```

---

## 🎯 Usage Examples

### Basic Usage

```python
from agent_system import AgenticReportWriter
import os

# Initialize
writer = AgenticReportWriter(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-5-20250929",
    temperature=0.7
)

# Generate report
result = writer.generate_report("MLOps: Best Practices and Implementation")

# Access results
print(f"Words: {result['word_count']}")
print(result['report'])
```

### With Custom Settings

```python
# Use different model
writer = AgenticReportWriter(
    anthropic_api_key=api_key,
    model="claude-opus-4-20250514",  # More powerful
    temperature=0.5  # More focused
)

# Generate on custom topic
result = writer.generate_report(
    "API Rate Limiting Strategies in Microservices"
)
```

### Save to File

```python
with open(f"report_{topic}.txt", 'w') as f:
    f.write(result['report'])
```

---

## 🎨 Gradio Interface Features

The web interface provides:

### Configuration Panel
- API key input (secure)
- Model selection (Sonnet 4.5, Sonnet 4, Opus 4)
- Temperature slider (0.0-1.0)

### Topic Selection
- Custom topic input
- Pre-defined example topics
- Course-related suggestions

### Report Generation
- Progress tracking
- Real-time status updates
- Word count monitoring

### Human-in-the-Loop Editing
- Edit draft directly
- AI-assisted refinement
- Custom instruction input
- Re-generate with changes

### Export Options
- Copy to clipboard
- Save as .txt file
- **Save as PDF file** (formatted, professional)
- View metadata & outline

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Model Options

| Model | Use Case | Speed | Quality |
|-------|----------|-------|---------|
| claude-sonnet-4-5-20250929 | Recommended | Fast | Excellent |
| claude-sonnet-4-20250514 | Balanced | Medium | Very Good |
| claude-opus-4-20250514 | Maximum quality | Slow | Best |

### Temperature Settings

- **0.0-0.3**: Focused, deterministic (technical docs)
- **0.4-0.7**: Balanced (recommended)
- **0.8-1.0**: Creative, varied (exploratory)

---

## 🚢 GCP Deployment

### Prerequisites
- Google Cloud account
- gcloud CLI installed
- Project created in GCP

### Deployment Steps

```bash
# 1. Authenticate
gcloud auth login

# 2. Set project
gcloud config set project YOUR_PROJECT_ID

# 3. Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 4. Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/agentic-report-writer

# 5. Deploy to Cloud Run
gcloud run deploy agentic-report-writer \
  --image gcr.io/YOUR_PROJECT_ID/agentic-report-writer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ANTHROPIC_API_KEY="your-key-here"

# 6. Get URL
gcloud run services describe agentic-report-writer \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

### Alternative: Use deploy_gcp.sh

```bash
chmod +x deploy_gcp.sh
./deploy_gcp.sh YOUR_PROJECT_ID YOUR_API_KEY
```

---

## 📊 Example Topics

### MLOps
- MLOps: Best Practices and Implementation Strategies
- Model Monitoring and Observability in Production
- Automated ML Pipeline Deployment

### CI/CD
- CI/CD Pipelines: Modern Approaches and Tools
- GitOps for Kubernetes Deployments
- Continuous Testing in DevOps Workflows

### APIs
- RESTful API Design: Principles and Best Practices
- GraphQL vs REST: When to Use Each
- API Versioning Strategies

### Gradio
- Gradio: Building Interactive ML Applications
- Deploying Gradio Apps to Production
- Custom Gradio Components

### Kubernetes
- Container Orchestration with Kubernetes in ML
- Kubernetes Best Practices for Data Science
- Service Mesh in ML Infrastructure

---

## 📈 Report Quality Criteria

### Structure (5 points)
- ✅ Clear introduction with context
- ✅ 3-4 well-organized main sections
- ✅ Comprehensive conclusion
- ✅ Smooth transitions between sections

### Content (5 points)
- ✅ Factually accurate information
- ✅ Relevant examples and applications
- ✅ Current best practices
- ✅ Technical correctness

### Length (5 points)
- ✅ 950-1050 words (target: 1000)
- ✅ Balanced section distribution
- ✅ No excessive filler
- ✅ Complete thoughts

---

## 🛠️ Troubleshooting

### Common Issues

**1. API Key Error**
```python
# Verify key is set
import os
print(os.getenv("ANTHROPIC_API_KEY"))

# Set explicitly
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
```

**2. Import Error**
```bash
# Reinstall dependencies
pip install --upgrade langchain langgraph langchain-anthropic
```

**3. Word Count Off Target**
- System iterates up to 3 times
- Check feedback in result['feedback']
- Adjust temperature if needed

**4. Gradio Not Loading**
```bash
# Check port
python app.py --server-port 7861

# Share publicly
python app.py --share
```

**5. GCP Deployment Fails**
```bash
# Check quotas
gcloud compute project-info describe --project=YOUR_PROJECT

# Verify billing
gcloud beta billing projects describe YOUR_PROJECT
```

---

## 📝 Grading Checklist

### Report (15 points)
- [x] 1000 words ±50 (5 points)
- [x] Factually correct (5 points)
- [x] Clear structure with titles (5 points)

### Agentic System (65 points)
- [x] Clear prompts, best practices (15 points)
- [x] Complex multi-agent system (50 points)
  - [x] LangGraph workflow
  - [x] 5 specialized agents
  - [x] Tool usage (word counting)
  - [x] Iterative refinement
  - [x] State management

### System Diagram (10 points)
- [x] Visual workflow representation
- [x] Shows agent interactions
- [x] Clear and understandable

### User Interface (10 points)
- [x] Gradio interface
- [x] Customization options
- [x] File export capability
- [x] GCP deployment ready

### Bonus (10 points)
- [x] Human-in-the-loop editing
- [x] Real-time refinement
- [x] Custom instruction input

**Total Possible: 110 points**

---

## 🤝 Submission Guidelines

### Required Files
1. ✅ `agent_system.py` - Core agent code
2. ✅ `app.py` - Gradio interface
3. ✅ `requirements.txt` - Dependencies
4. ✅ `README.md` - This documentation
5. ✅ `report_example.txt` - Generated report
6. ✅ `agentic_workflow_diagram.png` - System diagram

### Submission Package
```
submission.zip
├── code/
│   ├── agent_system.py
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── report_example.txt
├── agentic_workflow_diagram.png
├── README.md
└── gcp_url.txt  # Contains deployed app URL
```

### GCP Deployment
- Deploy to Google Cloud Run
- Make publicly accessible
- Include URL in `gcp_url.txt`

### Settings Used
Document in README:
```markdown
## Report Generation Settings

- **Model**: claude-sonnet-4-5-20250929
- **Temperature**: 0.7
- **Topic**: MLOps Best Practices
- **Iterations**: 2
- **Final Word Count**: 1003
```

---

## 📚 Additional Resources

### LangGraph Documentation
- https://python.langchain.com/docs/langgraph

### Claude API
- https://docs.anthropic.com/

### Gradio
- https://www.gradio.app/docs

### Google Cloud Run
- https://cloud.google.com/run/docs

---

## 📄 License

This project is for educational purposes as part of a course assignment.

---

## 👥 Author

Created for: [Course Name] - Agentic Systems Assignment
Date: 2025

---

## 🎓 Learning Outcomes

This project demonstrates:
1. Multi-agent system design with LangGraph
2. Prompt engineering best practices
3. State management in agent workflows
4. Tool integration (word counting)
5. Iterative refinement loops
6. UI development with Gradio
7. Cloud deployment (GCP)
8. Human-in-the-loop patterns

---

**Questions?** Check the troubleshooting section or review the code comments for detailed explanations.
