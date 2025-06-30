# OutScan Architecture Diagrams

This directory contains Mermaid source files and generated **high-definition** static images for all OutScan architecture diagrams.

> **🎯 Quality Note:** All PNG images are generated with **3x scale factor** for crisp, readable text at any zoom level. Perfect for presentations, documentation, and judge review!

## 📁 File Organization

### 1. High-Level Architecture (Conceptual Flow)
- **Source:** `1-high-level-architecture.mmd`
- **PNG:** `1-high-level-architecture.png` (361KB) - **High Definition**
- **SVG:** `1-high-level-architecture.svg` (28KB)

Shows the conceptual data flow from ingestion through processing, storage, alerting, and presentation.

### 2. AWS Serverless Infrastructure (Technical Details)
- **Source:** `2-aws-infrastructure.mmd`
- **PNG:** `2-aws-infrastructure.png` (176KB) - **High Definition**
- **SVG:** `2-aws-infrastructure.svg` (45KB)

Detailed technical diagram showing all AWS services, connections, and data flows.

### 3. Real-Time Processing Pipeline (Sequence Diagram)
- **Source:** `3-data-flow-sequence.mmd`
- **PNG:** `3-data-flow-sequence.png` (235KB) - **High Definition**
- **SVG:** `3-data-flow-sequence.svg` (43KB)

Step-by-step sequence showing the complete data processing workflow with 33 numbered steps.

## 🔧 Generation Commands

These images were generated using [mermaid-cli](https://github.com/mermaid-js/mermaid-cli):

```bash
# Install mermaid-cli globally
npm install -g @mermaid-js/mermaid-cli

# Generate HIGH-DEFINITION PNG images (3x scale, dark theme, transparent background)
mmdc -i 1-high-level-architecture.mmd -o 1-high-level-architecture.png -t dark -b transparent -s 3
mmdc -i 2-aws-infrastructure.mmd -o 2-aws-infrastructure.png -t dark -b transparent -s 3
mmdc -i 3-data-flow-sequence.mmd -o 3-data-flow-sequence.png -t dark -b transparent -s 3

# Generate HIGH-DEFINITION SVG images (3x scale, dark theme, transparent background)
mmdc -i 1-high-level-architecture.mmd -o 1-high-level-architecture.svg -t dark -b transparent -s 3
mmdc -i 2-aws-infrastructure.mmd -o 2-aws-infrastructure.svg -t dark -b transparent -s 3
mmdc -i 3-data-flow-sequence.mmd -o 3-data-flow-sequence.svg -t dark -b transparent -s 3
```

## 📋 Usage Recommendations

### For GitHub/GitLab README
Use the live Mermaid code blocks for interactive rendering:
```markdown
```mermaid
<!-- paste contents of .mmd file -->
```
```

### For Documentation Sites
Use SVG images for crisp scaling:
```markdown
![Architecture](./diagrams/1-high-level-architecture.svg)
```

### For Presentations/Reports
Use PNG images for maximum compatibility:
```markdown
![Architecture](./diagrams/1-high-level-architecture.png)
```

### For Print/PDF
Use SVG images which scale perfectly without pixelation.

## 🎨 Styling

All diagrams use consistent AWS-branded styling:
- **AWS Services:** Orange (#FF9900)
- **Lambda Functions:** Orange (#FF9900) 
- **Storage Services:** Blue (#2196F3)
- **Monitoring:** Purple (#9C27B0)
- **External Systems:** Green (#4CAF50)
- **Processing:** Green (#4CAF50)
- **Alerting:** Red (#F44336)

## 🔄 Regenerating Images

To regenerate all images after modifying source `.mmd` files:

```bash
# From the OutScan root directory
cd diagrams

# Regenerate all HIGH-DEFINITION PNG images (3x scale)
for file in *.mmd; do
    mmdc -i "$file" -o "${file%.mmd}.png" -t dark -b transparent -s 3
done

# Regenerate all HIGH-DEFINITION SVG images (3x scale)
for file in *.mmd; do
    mmdc -i "$file" -o "${file%.mmd}.svg" -t dark -b transparent -s 3
done
```

---

**📋 Note:** These diagrams are production-ready and tested across multiple Mermaid renderers for maximum compatibility. 