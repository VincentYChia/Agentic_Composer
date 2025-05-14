Project Overview: Maestro AI - Sheet Music Generation Research
Project Summary
This research project explores the development and capabilities of Maestro, an AI system specifically designed to generate musical sheet music in MusicXML format. While numerous AI systems can generate audio music, Maestro uniquely focuses on producing properly formatted sheet music for instrumental orchestrations such as full orchestra, string quartet, concert band, and brass quintet. The system was trained on classical music compositions and employs a multi-stage approach to generate playable, structured compositions.
Research Objectives

To develop an AI capable of generating proper sheet music notation (not just MIDI or audio)
To evaluate the quality of AI-generated compositions using 1/f pitch and duration metrics as quantitative benchmarks
To create a tool that can help musicians, particularly beginners, with composition

Methodology
Data Collection and Processing

Source material: Approximately 2,000 classical music scores from IMSLP (International Music Score Library Project)
Processing workflow:

PDF score collection from IMSLP (copyright-free compositions)
Conversion to MusicXML format using Audiveris (accelerated with AWS)
Final training dataset: 495 XML files plus 5 "reasoned out" pieces with explanations for note choices and rhythms



Model Architecture and Training

Base model: Fine-tuned version of OpenAI's o4mini model
Multi-stage agentic approach:

Initial planner model (o4mini-finetuned): Conceptualizes the composition
Outline refinement (GPT-4.1): Develops the initial musical concept
Second refinement and formatting (GPT-4.1): Structures the outline for XML generation
XML writer (o4mini-finetuned): Produces the final MusicXML code


Training process: Supervised learning with fine-tuning

Data preparation took months, while actual model training completed in under an hour



Evaluation Metrics

Primary quantitative benchmark: 1/f pitch and potentially 1/f duration analysis

Maestro-generated compositions scored between 1.1-1.3
Classical composers typically score around 0.8-0.9
This difference indicates Maestro produces more structured and repetitive compositions


Playability: All compositions verified to be technically playable
No formal human evaluation due to the subjective nature of musical quality

Key Findings and Results

Successfully created an AI system capable of generating valid MusicXML sheet music
The system performs well with moderately complex prompts and can handle various orchestration types
Output analysis shows high structure and repeatability compared to human compositions
System demonstrates competence in note and rhythm generation while struggling with some musical elements

System Capabilities and Limitations
Capabilities

Generates complete sheet music in MusicXML format
Handles various orchestration types (full orchestra, chamber ensembles, etc.)
Works well with general prompts about mood, style, and instrumentation
Produces technically playable compositions
Outputs can be viewed and edited in standard notation software like Sibelius

Limitations

Performs better with general prompts than highly specific directions
More suitable for beginners than advanced composers with precise requirements
Struggles with some musical elements/markings beyond notes and rhythms
Some musical elements don't render properly in Sibelius

User Interaction
Users can interact with Maestro through simple prompts that specify:

Desired mood or theme
Instrumentation
Length (typically around 40 measures but flexible)
Basic stylistic guidance

Example prompt: "Write me a composition that captures the green spaces of the world. Make it light and peaceful. Full orchestra 40 measures"
Implementation Details

Website with API integration allows users to access the system
Technical workflow implements sequential processing through the four model stages
Extensive prompt engineering was required to encourage creativity while maintaining proper MusicXML structure

Future Directions

Scale to larger models to improve handling of musical elements and creativity
Expand training data to include more diverse musical styles
Improve the rendering of musical elements in notation software
Further refinement of the multi-stage approach
