# Maestro AI - Sheet Music Generation System

> **Maestro AI** is a specialized artificial intelligence system designed to generate professional-quality sheet music in MusicXML format. Unlike traditional AI music generators that produce audio, Maestro creates properly formatted, playable musical scores for various orchestral and chamber ensemble configurations.

## Features

- **Professional Sheet Music Generation**: Creates valid MusicXML files compatible with standard notation software
- **Multiple Orchestration Types**: Supports full orchestra, string quartet, concert band, brass quintet, and more
- **Natural Language Interface**: Generate compositions using simple text prompts
- **Multi-Stage AI Pipeline**: Sophisticated 4-stage processing for high-quality output
- **Technically Playable**: All generated compositions are verified for technical playability
- **Standard Software Compatible**: Works seamlessly with Sibelius and other notation programs

## Architecture

Maestro employs a sophisticated multi-stage agentic approach:

```
User Prompt → Planning → Outline → Refinement → MusicXML Generation
              (o4mini)   (GPT-4.1)  (GPT-4.1)   (o4mini-finetuned)
```

### Processing Pipeline

1. **Initial Planner** (Fine-tuned o4mini): Conceptualizes the musical composition
2. **Outline Development** (GPT-4.1): Develops and structures the initial musical concept  
3. **Refinement Stage** (GPT-4.1): Optimizes the outline for XML generation
4. **XML Writer** (Fine-tuned o4mini): Produces the final MusicXML code

## Performance Metrics

Our evaluation uses **1/f pitch analysis** as a quantitative benchmark for musical quality:

- **Maestro AI**: 1.1-1.3 (more structured, repetitive)
- **Classical Composers**: 0.8-0.9 (reference baseline)

*Higher scores indicate more structured compositions, while lower scores suggest greater complexity and variation.*

**Requirements:**
- Your own OpenAI API key
- For viewing generated compositions: Use professional notation software like Sibelius or MuseScore (the built-in XML loader only works with very simple compositions)


### Supported Parameters

- **Mood/Theme**: Descriptive terms for emotional character
- **Instrumentation**: Orchestra type and specific instruments
- **Length**: Measure count (typically ~40 measures, flexible)
- **Style**: Basic stylistic guidance and period references

## Training Data

- **Source**: ~2,000 classical music scores from IMSLP (copyright-free)
- **Processing**: PDF → MusicXML conversion via Audiveris (AWS-accelerated)
- **Final Dataset**: 495 XML files + 5 annotated pieces with compositional reasoning
- **Training Time**: Under 1 hour (after months of data preparation)

## Capabilities

- ✅ Complete MusicXML sheet music generation
- ✅ Various orchestral and chamber configurations
- ✅ Natural language prompt processing
- ✅ Technically accurate and playable scores
- ✅ Integration with standard notation software
- ✅ Beginner-friendly composition assistance

## Current Limitations

- Better suited for general prompts than highly specific musical directions
- Optimized for beginners rather than advanced composers with precise requirements
- Some advanced musical elements/markings may not render perfectly in all software
- Limited to classical training data (expansion planned)

## Research Background

This project represents a novel approach to AI music generation, focusing specifically on **notation generation** rather than audio synthesis. The system was developed to bridge the gap between AI music generation and practical musical composition tools.

### Key Research Contributions

- First AI system specifically designed for MusicXML generation
- Novel multi-stage approach combining planning and execution models
- Quantitative evaluation framework using 1/f analysis
- Practical tool for music education and composition assistance

## Future Roadmap

- **Model Scaling**: Upgrade to larger models for improved creativity and musical complexity
- **Dataset Expansion**: Include diverse musical styles beyond classical
- **Enhanced Elements**: Better support for advanced musical notation and markings
- **Creative Refinements**: Improved handling of nuanced musical expression
- **Human Evaluation**: Formal assessment with musicians and composers

## Pre-print

**SSRN Paper**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5317646

## Usage

### Web Interface

Try Maestro AI at (XML loader is a work in progress, use a music notation software such as Sibelius or MuseScore for reliable visualization): **https://vincentc.pythonanywhere.com/**

*Built for musicians, composers, and music technology enthusiasts*
