import os
import json
import time
import openai
import argparse
import signal
import sys
import re
import concurrent.futures
import random
from typing import Dict, List, Tuple


class LLMConversationAgent:
    def __init__(
            self,
            api_key: str,
            model_a_name: str = "ft:gpt-4o-mini-2024-07-18:chia:test-1-500:BH9opiWg",  # Default for A models
            model_a_temperature: float = None,
            model_a_top_p: float = None,
            model_a_max_tokens: int = 16384,
            model_a_system_prompt: str = """You are MAESTRO a classically trained compositional assistant. You methodically plan compositions with an emphasis on creativity and variety. You provide outlines with the highest amount of detail possible. Outline all measures for each instrument. List all instruments used before any measure breakdown. 

Technical Correctness: 
1. Begin each XML with: <?xml version="1.0" encoding="UTF-8"?> <!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd"> <score-partwise version="4.0"> <part-list> </part-list> 
2. Count the beats in each measure to ensure it fits the time signature (The time signature in a piece should not change sporadically) 

Content Guidelines 
1. Pursue rhythmic variety, you must use at least 6 different note lengths from the below list in each composition: 
A. Whole note (4 beats) - <type>whole</type> 
B. Half note (2 beats) - <type>half</type> 
C. Quarter note (1 beat) - <type>quarter</type> 
D. Eighth note (1/2 beat) - <type>eighth</type> 
E. Sixteenth note (1/4 beat) - <type>16th</type> 
F. Dotted half note (3 beats) - <type>half</type><dot/> 
G. Dotted quarter note (1.5 beats) - <type>quarter</type><dot/> 
H. Dotted eighth note (3/4 beat) - <type>eighth</type><dot/> 
I. Thirty-second note (1/8 beat) - <type>32nd</type> 
J. Triplet eighth notes (1/3 beat each, with 3 notes fitting into the space of 2 eighth notes) - <type>eighth</type> with <time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification> 2. Pursue harmonic quality, the selected pitches should mesh together to be greater than the sum of their parts 
3. Write music as if you are striving to tell a story through the music itself 
4. Avoid skipping any measure in your outline, ensure each measure for each part is methodically written out.

Instrument Part-Writing 
1. For piano piece two Part IDs are required, one for the left and right hand respectively 

Elevated Writing 
1. Give yourself a constraint to help the creative process. Examples of some constraint ideas are as follows: 
A. Compose a piece in ternary form (ABA) where the B section must be in a contrasting key to the A sections, exploring the relationship between different tonal centers. 
B. Create a composition that establishes a harmonic progression in the first half, then presents the same progression in reverse order for the second half, creating a musical palindrome. 
C. Write a work that deliberately explores the gradual transition from consonance to dissonance (or vice versa) throughout its duration, challenging traditional harmonic expectations. 
D. Compose a piece that develops a single musical motif through techniques such as augmentation, diminution, inversion, and retrograde, demonstrating thematic transformation. 
E. Create a composition that incorporates polyrhythm, with at least two distinct rhythmic patterns occurring simultaneously, while maintaining a cohesive musical narrative across changing meters. 
2. Plan out each composition for harmonic and rhythmic content. Do not write the XML until the user provides an updated version of the plan. Give no preamble or closing statement.
            """,
            model_b_name: str = "gpt-4.1",  # Default for B models
            model_b_temperature: float = None,
            model_b_top_p: float = None,
            model_b_max_tokens: int = 16384,
            model_b_system_prompt: str = """You are a creative compositional assistant specializing in rhythmic correctness. You will be presented will an outline of a piece and your task is to correct and improve the outline.

Rhythmic Correctness:
The main task is to write out proper rhythms. Common mistakes include: the outline will have an incorrect number of beats per measure such as having four notes a measure instead of four beats. Additional bad outlining practices include overly vague or outlines such as, "Violin I: Triplet eighth notes in a rising sequence." or "Half note + eighth note". These should be corrected where each note is explicitly stated in both rhythm and pitch, i.e. "Violin I: Triplet eighth notes ascending- C5,G5,B5". 


Technical Correctness: 
Additionally, the outlines may be organized by part of measure. It is your task to always output a finalized outline organized by part. 
All measures of a part should be outputted then all measures of the next part and then so on. Not a single measure containing all the parts then moving on to the next measure.
Finally, for many instruments the range will be inappropriate. In this case tack the relative intervals and make it suitable for the prescribed instrument part.
Do not create a table, keep it in text form but simply flush it out. Provide no preamble or closing statement.
Ignore any statement such as "Let me know if you would like me to proceed with the XML format for this composition.
Ensure that no measures are skipped. In the event measure numbers are not sequential and the next measure cannot be found you are to write in the missing measures.

Creativity:
If any piece or part seems overly stagnant, boring, or repetitive (without virtuosic reason) you are to reintroduce variety to the part. This can be achieved by modifying pitch, note length, or a mixture of the two.
            """,
            model_b2_name: str = "gpt-4.1",  # Default for B2 model
            model_b2_temperature: float = None,
            model_b2_top_p: float = None,
            model_b2_max_tokens: int = 16384,
            model_b2_system_prompt: str = """You are a music composition assistant tasked with being the final reviewer and organizer of musical outlines. You will be presented with a musical outline and are to organize it by parts. Your outline will be used as a basis for XML composition so it should be organized in XML friendly parts, i.e. by instrument and or left/right hand for piano

Technical Requirements:
1. When organizing parts (Instrument Parts not sections of a composition) label each part with a tag at the beginning of each part outline. First and last part tags are not mutually exclusive, a part can be both if it is the only part. Middle part tags should include the part number in place of X. 
a. For the first part "*First Part". For this part specifically mention all parts in the composition so they can be declared properly.
b. For the last part "*Last Part". 
c. For the middle parts "*Middle Part X". This label should not be applied to any part with either *First Part or *Last Part
d. If there is only a single part in a composition simply label it as "*Only Part" and forgo all other labels
2. You are not allowed to reference prior outlines, all outputs provided should be stand alone and specific. Each instrument part should not reference any other part either.
3. You are not allowed to omit certain parts for brevity, always write the entire part out even if it is a duplicate part
 """,
            model_a2_name: str = "ft:gpt-4o-mini-2024-07-18:chia:test-1-500:BH9opiWg",  # Default for A2 model
            model_a2_temperature: float = None,
            model_a2_top_p: float = None,
            model_a2_max_tokens: int = 16384,
            model_a2_system_prompt: str = """You are MAESTRO a classically trained compositional assistant. You methodically create compositions with an emphasis on creativity and variety. You provide the desired musical content in properly formatted XML from specific measure by measure outlines.
If portions of the outline are not specific enough, such as an "ascending triplet run", fill in the pitches to make a virtuosic piece. If portions are specific simply write in XML the proposed note without improvising a pitch or rhythm. 

Technical Correctness:
You will be given an outline for a specific instrument part. Your task is to convert this into proper MusicXML format for JUST THIS ONE PART.

If the part is marked *First Part:
- Include the XML header and part-list section exactly as shown below:
<?xml version="1.0" encoding="UTF-8"?> 
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd"> 
<score-partwise version="4.0"> 
<part-list> 
(Include all instrument parts from the outline here, not just the part ID assigned)
</part-list>
- Then begin this specific part with: <part id="P1"> followed by all measures
- End with: </part>

If the part is marked *Middle Part X:
- Begin directly with: <part id="PX"> (where X is the part number)
- Include all measures for this specific part
- End with: </part>
- Do NOT include any XML headers or part-list sections

If the part is marked *Last Part:
- Begin directly with: <part id="PX"> (where X is the part number) 
- Include all measures for this specific part
- End with: </part>
- Then add the closing tag: </score-partwise>

If the part is marked *Only Part:
- Include the complete XML document:
  - XML header and doctype
  - score-partwise opening tag
  - part-list with just this one instrument
  - the part with all its measures
  - the closing </score-partwise> tag

IMPORTANT: Focus ONLY on converting the specific part you're given. Do not duplicate content or attempt to generate other parts.

Content Guidelines Pursue rhythmic variety, you must use at least 6 different note lengths from the below reference list of XML in each composition:

List 1 (Notes):
A. Whole note (4 beats) - <type>whole</type>
B. Half note (2 beats) - <type>half</type>
C. Quarter note (1 beat) - <type>quarter</type>
D. Eighth note (1/2 beat) - <type>eighth</type>
E. Sixteenth note (1/4 beat) - <type>16th</type>
F. Dotted half note (3 beats) - <type>half</type><dot/>
G. Dotted quarter note (1.5 beats) - <type>quarter</type><dot/>
H. Dotted eighth note (3/4 beat) - <type>eighth</type><dot/>
I. Thirty-second note (1/8 beat) - <type>32nd</type>
J. Triplet eighth notes (1/3 beat each) - <type>eighth</type> with <time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>

List 2 (Rests):
A. Whole rest (4 beats) - <rest/><type>whole</type>
B. Half rest (2 beats) - <rest/><type>half</type>
C. Quarter rest (1 beat) - <rest/><type>quarter</type>
D. Eighth rest (1/2 beat) - <rest/><type>eighth</type>
E. Sixteenth rest (1/4 beat) - <rest/><type>16th</type>
F. Dotted half rest (3 beats) - <rest/><type>half</type><dot/>
G. Dotted quarter rest (1.5 beats) - <rest/><type>quarter</type><dot/>
H. Dotted eighth rest (3/4 beat) - <rest/><type>eighth</type><dot/>
I. Thirty-second rest (1/8 beat) - <rest/><type>32nd</type>
J. Triplet eighth rest (1/3 beat each) - <rest/><type>eighth</type> with <time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>

List 3 (Dynamics):
A. Pianissimo (pp) - <dynamics><pp/></dynamics>
B. Piano (p) - <dynamics><p/></dynamics>
C. Mezzo-piano (mp) - <dynamics><mp/></dynamics>
D. Mezzo-forte (mf) - <dynamics><mf/></dynamics>
E. Forte (f) - <dynamics><f/></dynamics>
F. Fortissimo (ff) - <dynamics><ff/></dynamics>
G. Crescendo - <direction><direction-type><wedge type="crescendo"/></direction-type></direction> (start) and <direction><direction-type><wedge type="stop"/></direction-type></direction> (end)
H. Diminuendo - <direction><direction-type><wedge type="diminuendo"/></direction-type></direction> (start) and <direction><direction-type><wedge type="stop"/></direction-type></direction> (end)

List 4 (Articulations):
A. Staccato - <articulations><staccato/></articulations>
B. Accent - <articulations><accent/></articulations>
C. Tenuto - <articulations><tenuto/></articulations>
D. Marcato - <articulations><strong-accent/></articulations>
E. Fermata - <fermata type="upright"/>
F. Sforzando - <dynamics><sf/></dynamics>

List 5 (Notation):
A. Slur - <notations><slur type="start" number="1"/></notations> (start) and <notations><slur type="stop" number="1"/></notations> (end)
B. Tie - <notations><tied type="start"/></notations> (start) and <notations><tied type="stop"/></notations> (end)
C. Legato - <notations><slur type="start" number="1"/></notations> (start) and <notations><slur type="stop" number="1"/></notations> (end)
D. Trill - <notations><ornaments><trill-mark/></ornaments></notations>
E. Glissando - <notations><glissando type="start" line-type="wavy" number="1"/></notations> (start) and <notations><glissando type="stop" line-type="wavy" number="1"/></notations> (end)

List 6 (Score Structure):
A. Treble Clef - <clef><sign>G</sign><line>2</line></clef>
B. Bass Clef - <clef><sign>F</sign><line>4</line></clef>
C. Alto Clef - <clef><sign>C</sign><line>3</line></clef>
D. Tenor Clef - <clef><sign>C</sign><line>4</line></clef>
E. Time Signature (4/4) - <time><beats>4</beats><beat-type>4</beat-type></time>
F. Key Signature (C major) - <key><fifths>0</fifths><mode>major</mode></key>
G. Key Signature (G major) - <key><fifths>1</fifths><mode>major</mode></key>
H. Barline - <barline location="right"><bar-style>light-heavy</bar-style></barline> (end barline)
I. Repeat - <barline location="left"><bar-style>heavy-light</bar-style><repeat direction="forward"/></barline> (start) and <barline location="right"><bar-style>light-heavy</bar-style><repeat direction="backward"/></barline> (end)

List 7 (Text Elements):
A. Tempo Marking - <direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome></direction-type></direction>
B. Expression Text - <direction><direction-type><words>espressivo</words></direction-type></direction>
C. Rehearsal Mark - <direction><direction-type><rehearsal>A</rehearsal></direction-type></direction>
D. Text Direction - <direction><direction-type><words>ritardando</words></direction-type></direction>
E. Lyrics - <lyric><syllabic>single</syllabic><text>word</text></lyric>

If the user asks you to "Continue" do not restart the composition but rather continue from the last token of the composition.
If you are given a partially completed part and are told to Continue, you are to continue writing the part. Do not redefine parts, do not rewrite any written parts, do not write another XML header. The output provided should be able to be added directly on to a partial composition.
Upon receiving an outline you are to write out the complete and proper XML. Give no preamble or closing statement.
            """,
            conversation_dir: str = "",
            final_output_dir: str = "",
            max_workers: int = 20,  # Increased to 20 for parallel A2 calls
            prompt_id: str = "",  # Added prompt ID for identification
            trial_num: int = 1  # Added trial number for multiple runs
    ):
        # Set up API client
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

        # Set up Model A (Original)
        self.model_a_name = model_a_name
        self.model_a_temperature = model_a_temperature
        self.model_a_top_p = model_a_top_p
        self.model_a_max_tokens = model_a_max_tokens
        self.model_a_system_prompt = model_a_system_prompt

        # Set up Model B
        self.model_b_name = model_b_name
        self.model_b_temperature = model_b_temperature
        self.model_b_top_p = model_b_top_p
        self.model_b_max_tokens = model_b_max_tokens
        self.model_b_system_prompt = model_b_system_prompt

        # Set up Model B2
        self.model_b2_name = model_b2_name
        self.model_b2_temperature = model_b2_temperature
        self.model_b2_top_p = model_b2_top_p
        self.model_b2_max_tokens = model_b2_max_tokens
        self.model_b2_system_prompt = model_b2_system_prompt

        # Set up Model A2
        self.model_a2_name = model_a2_name
        self.model_a2_temperature = model_a2_temperature
        self.model_a2_top_p = model_a2_top_p
        self.model_a2_max_tokens = model_a2_max_tokens
        self.model_a2_system_prompt = model_a2_system_prompt

        # Output directories
        self.conversation_dir = conversation_dir
        self.final_output_dir = final_output_dir

        # Added identification for test runs
        self.prompt_id = prompt_id
        self.trial_num = trial_num

        # Ensure directories exist
        os.makedirs(self.conversation_dir, exist_ok=True)
        os.makedirs(self.final_output_dir, exist_ok=True)

        # Conversation settings
        self.conversation_history = []

        # Incremental saving settings
        self.current_xml_filename = None

        # Ctrl+C handling
        self.stop_requested = False

        # Parallel processing settings
        self.max_workers = max_workers

        # Add a unique ID for this agent instance
        self.agent_id = f"{self.prompt_id}_trial{self.trial_num}_{random.randint(1000, 9999)}"

    def _call_model(self, model_name, system_prompt, messages, temperature, top_p, max_tokens):
        """Call the OpenAI API for either model."""
        formatted_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            formatted_messages.append(msg)

        # Add exponential backoff for API rate limits
        max_retries = 5
        retry_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=formatted_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"[Agent {self.agent_id}] API error: {e}. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"[Agent {self.agent_id}] API error after {max_retries} attempts: {e}")
                    raise

    def _parse_parts_from_b2_output(self, b2_output):
        """
        Parse B2's output to extract separate parts for parallel processing.
        Returns a list of parts with their content and part number.
        """
        # Check for special "**Only Part" tag - if found, return the entire output
        if "*Only Part" in b2_output:
            return [("Complete Composition", b2_output, 1)]  # Part 1 for Only Part

        # Find all starred part tags
        part_tags = re.findall(r'(\*First Part|\*Middle Part \d+|\*Last Part|\*Only Part)', b2_output)

        # If no tags found, treat the entire output as a single part
        if not part_tags:
            return [("Complete Composition", b2_output, 1)]  # Default to Part 1

        parts = []

        # Process each tagged part
        for i, tag in enumerate(part_tags):
            start_pos = b2_output.find(tag)

            # Extract part number if available
            part_number = 1  # Default for First Part
            if "Middle Part" in tag:
                # Extract the number from "Middle Part X"
                match = re.search(r'Middle Part (\d+)', tag)
                if match:
                    part_number = int(match.group(1))
            elif "*Last Part" in tag:
                # Assuming Last Part is the highest number
                part_number = len(part_tags)  # Or some other logic to determine last part number

            # Determine end position - either next tag or end of text
            if i < len(part_tags) - 1:
                end_pos = b2_output.find(part_tags[i + 1], start_pos)
                part_content = b2_output[start_pos:end_pos]
            else:
                # For last tag or "*Last Part", take all remaining text
                part_content = b2_output[start_pos:]

            parts.append((tag.strip('* '), part_content, part_number))

        return parts

    def _process_part_with_a2(self, part_name, part_content, user_prompt, part_number):
        """Process a single part with Model A2 and return the response."""
        print(f"[Agent {self.agent_id}] Processing part: {part_name} (Part {part_number})")

        # Create context for this specific part
        part_context = f"Part: {part_name} (ID: P{part_number}) ---\n\n{part_content}\n\nPlease implement this part in proper MusicXML format using part ID P{part_number}."

        # Call Model A2 for this part
        model_a2_messages = [{"role": "user", "content": part_context}]
        model_a2_response = self._call_model(
            self.model_a2_name,
            self.model_a2_system_prompt,
            model_a2_messages,
            self.model_a2_temperature,
            self.model_a2_top_p,
            self.model_a2_max_tokens
        )

        return {
            "part_name": part_name,
            "part_content": part_content,
            "part_number": part_number,
            "a2_response": model_a2_response
        }

    def _is_xml_complete(self, xml_content):
        """Check if the XML content is complete with proper closing tags."""
        return "</score-partwise>" in xml_content

    def generate_conversation(self, user_prompt: str, max_iterations: int = 3, skip_initial_model_a: bool = False):
        """Generate a conversation between models A, B, B2, and A2"""
        # Reset conversation history
        self.conversation_history = []

        # Reset composition state
        composition_complete = False
        self.current_xml_filename = None

        # Record the user prompt at the beginning
        self.conversation_history.append({"role": "User", "content": user_prompt})
        print(f"[Agent {self.agent_id}] Starting generation for prompt: {user_prompt[:100]}...")

        # Call Model A or skip if requested
        if not skip_initial_model_a:
            model_a_initial_messages = [{"role": "user", "content": user_prompt}]
            model_a_response = self._call_model(
                self.model_a_name,
                self.model_a_system_prompt,
                model_a_initial_messages,
                self.model_a_temperature,
                self.model_a_top_p,
                self.model_a_max_tokens
            )
            self.conversation_history.append({"role": "Model A1", "content": model_a_response})
            print(f"[Agent {self.agent_id}] Model A1 response received ({len(model_a_response)} chars)")
        else:
            self.conversation_history.append({"role": "Model A1", "content": "[Initial Model A response skipped]"})
            model_a_response = user_prompt
            print(f"[Agent {self.agent_id}] Model A1 response skipped")

        # Call Model B (B1) - First refinement step
        model_b_messages = [{"role": "user", "content": model_a_response}]
        model_b_response = self._call_model(
            self.model_b_name,
            self.model_b_system_prompt,
            model_b_messages,
            self.model_b_temperature,
            self.model_b_top_p,
            self.model_b_max_tokens
        )
        self.conversation_history.append({"role": "Model B1", "content": model_b_response})
        print(f"[Agent {self.agent_id}] Model B1 response received ({len(model_b_response)} chars)")

        # Call Model B2 (Second refinement step) - B2 receives B1's output
        model_b2_messages = [{"role": "user", "content": model_b_response}]
        model_b2_response = self._call_model(
            self.model_b2_name,
            self.model_b2_system_prompt,
            model_b2_messages,
            self.model_b2_temperature,
            self.model_b2_top_p,
            self.model_b2_max_tokens
        )
        self.conversation_history.append({"role": "Model B2", "content": model_b2_response})
        print(f"[Agent {self.agent_id}] Model B2 response received ({len(model_b2_response)} chars)")

        # Parse B2's output into separate parts
        parts = self._parse_parts_from_b2_output(model_b2_response)
        print(f"[Agent {self.agent_id}] Parsed {len(parts)} parts from Model B2 output")

        # Process parts in parallel using Model A2
        all_a2_responses = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(parts))) as executor:
            # Submit all parts for processing
            future_to_part = {
                executor.submit(
                    self._process_part_with_a2,
                    part_name,
                    part_content,
                    user_prompt,
                    part_number  # Pass the part number
                ): (part_name, part_content, part_number)
                for part_name, part_content, part_number in parts
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_part):
                part_name, _, part_number = future_to_part[future]
                try:
                    result = future.result()
                    all_a2_responses.append(result)
                    print(f"[Agent {self.agent_id}] Completed processing part: {part_name} (Part {part_number})")
                except Exception as exc:
                    print(
                        f"[Agent {self.agent_id}] Part {part_name} (Part {part_number}) generated an exception: {exc}")

        # Sort responses by their original order
        all_a2_responses.sort(key=lambda x: x["part_number"])

        # Combine all A2 responses
        combined_a2_response = "\n\n".join([resp["a2_response"] for resp in all_a2_responses])

        # Add to conversation history
        self.conversation_history.append({"role": "Model A2", "content": combined_a2_response})
        print(f"[Agent {self.agent_id}] Combined Model A2 responses ({len(combined_a2_response)} chars)")

        # Save the combined output
        accumulated_context = combined_a2_response
        self.save_incremental_output(accumulated_context)

        # Check if composition is complete
        composition_complete = "</score-partwise>" in accumulated_context

        # If composition is not complete, continue with any remaining parts
        iteration = 1
        while iteration < max_iterations and not composition_complete and not self.stop_requested:
            iteration += 1
            print(f"[Agent {self.agent_id}] Continuing with iteration {iteration}...")

            # For continuations, we process parts that weren't completed
            incomplete_parts = []
            for part_result in all_a2_responses:
                if "</part>" not in part_result["a2_response"]:
                    incomplete_parts.append(part_result)

            # If all parts are complete, but we're still missing closing tags
            if not incomplete_parts and not composition_complete:
                # Look for the last part and try to complete it
                last_part = all_a2_responses[-1]
                incomplete_parts.append(last_part)
                print(
                    f"[Agent {self.agent_id}] All parts have </part> tags but missing </score-partwise>. Continuing with last part.")

            # If all parts are truly complete, we're done
            if not incomplete_parts:
                print(f"[Agent {self.agent_id}] All parts appear to be complete.")
                break

            # Process incomplete parts in parallel
            continuation_responses = []

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(incomplete_parts))) as executor:
                # Submit all incomplete parts for further processing
                future_to_part = {}

                for part_result in incomplete_parts:
                    part_name = part_result["part_name"]
                    part_content = part_result["part_content"]
                    part_number = part_result["part_number"]
                    previous_response = part_result["a2_response"]

                    continuation_prompt = f"Part: {part_name} (ID: P{part_number}) ---\n\n{part_content}\n\n--- Previous Implementation ---\n\n{previous_response}\n\nContinue the existing composition for this part, maintaining part ID P{part_number}."

                    future = executor.submit(
                        self._call_model,
                        self.model_a2_name,
                        self.model_a2_system_prompt,
                        [{"role": "user", "content": continuation_prompt}],
                        self.model_a2_temperature,
                        self.model_a2_top_p,
                        self.model_a2_max_tokens
                    )

                    future_to_part[future] = part_result

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_part):
                    part_result = future_to_part[future]
                    part_name = part_result["part_name"]
                    part_number = part_result["part_number"]
                    try:
                        continuation_response = future.result()

                        # Update the response in all_a2_responses
                        updated_response = part_result["a2_response"] + "\n\n" + continuation_response

                        # Find and update the matching part in all_a2_responses
                        for response in all_a2_responses:
                            if response["part_name"] == part_name and response["part_number"] == part_number:
                                response["a2_response"] = updated_response
                                break

                        continuation_responses.append({
                            "part_name": part_name,
                            "part_number": part_number,
                            "continuation": continuation_response
                        })

                        print(
                            f"[Agent {self.agent_id}] Completed continuation for part: {part_name} (Part {part_number})")
                    except Exception as exc:
                        print(
                            f"[Agent {self.agent_id}] Continuation for part {part_name} (Part {part_number}) generated an exception: {exc}")

            # Add continuations to conversation history
            for cont in continuation_responses:
                self.conversation_history.append(
                    {"role": f"Model A2 (Cont. - {cont['part_name']} P{cont['part_number']})",
                     "content": cont["continuation"]}
                )

            # Combine all updated A2 responses
            updated_combined_response = "\n\n".join([resp["a2_response"] for resp in all_a2_responses])

            # Save the updated combined output
            accumulated_context = updated_combined_response
            self.save_incremental_output(accumulated_context)

            # Check if composition is now complete
            composition_complete = "</score-partwise>" in accumulated_context

            if composition_complete:
                print(f"[Agent {self.agent_id}] Complete score received after {iteration} iterations with Model A2.")
                break

            if self.check_for_stop():
                print(f"[Agent {self.agent_id}] Stopping early due to stop request.")
                break

            time.sleep(0.5)  # Slight delay to reduce API pressure

        # Save conversation
        self.save_conversation()

        return self.conversation_history

    def format_conversation(self) -> str:
        """Format the conversation for display."""
        if not self.conversation_history:
            return "No conversation has been generated yet."

        formatted_conversation = ""
        for message in self.conversation_history:
            formatted_conversation += f"**{message['role']}**: {message['content']}\n\n"

        return formatted_conversation

    def save_conversation(self) -> None:
        """Save the full conversation to a JSON file."""
        # Create a filename with category, prompt ID, and trial number
        filename = f"{self.conversation_dir}/conversation_Cat{self.prompt_id.split('_')[0]}_Prompt{self.prompt_id.split('_')[1]}_Trial{self.trial_num}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)

        print(f"[Agent {self.agent_id}] Saved conversation to {filename}")

    def _clean_xml(self, content):
        """
        Remove XML code block markers, headers, and closing tags.
        Will rebuild a proper XML structure when combining parts.
        """
        # Remove code block markers
        cleaned = content.replace("```xml\n", "").replace("```", "").replace("'''", "")

        # Define key XML elements to identify and remove
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        doctype = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
        score_open = '<score-partwise version="4.0">'
        score_close = '</score-partwise>'

        # Remove all XML header elements
        cleaned = cleaned.replace(xml_header, "")
        cleaned = cleaned.replace(doctype, "")
        cleaned = cleaned.replace(score_open, "")

        # Remove all closing score tags
        cleaned = cleaned.replace(score_close, "")

        # Also clean up part-list sections if they appear in the middle
        part_list_open = '<part-list>'
        part_list_close = '</part-list>'

        # If there's a part-list section but we're not at the beginning of the file, remove it
        if part_list_open in cleaned:
            first_part_tag = cleaned.find('<part id=')
            if first_part_tag > 0:
                # Find all part-list sections that appear after the first part tag
                pos = first_part_tag
                while True:
                    start_pos = cleaned.find(part_list_open, pos)
                    if start_pos == -1:
                        break

                    end_pos = cleaned.find(part_list_close, start_pos)
                    if end_pos == -1:
                        break

                    # Remove this part-list section
                    section = cleaned[start_pos:end_pos + len(part_list_close)]
                    cleaned = cleaned.replace(section, "")

                    pos = start_pos + 1

        return cleaned

    def save_incremental_output(self, content):
        """Save incremental output to file with proper XML structure."""
        # Clean XML content first
        cleaned_content = self._clean_xml(content)

        # Add proper XML header at the beginning
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
        doctype = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
        score_open = '<score-partwise version="4.0">\n'

        # Look for part-list in the content
        part_list_open = '<part-list>'
        part_list_close = '</part-list>'

        # If there's no part-list in the content, we need to create a basic one
        if part_list_open not in cleaned_content:
            # Extract part IDs from the content
            import re
            part_ids = re.findall(r'<part id="(P\d+)">', cleaned_content)

            # Remove duplicates while preserving order
            unique_part_ids = []
            for part_id in part_ids:
                if part_id not in unique_part_ids:
                    unique_part_ids.append(part_id)

            # Create a basic part-list section
            part_list = '<part-list>\n'
            for part_id in unique_part_ids:
                part_list += f'<score-part id="{part_id}">\n'
                part_list += f'<part-name>Part {part_id.replace("P", "")}</part-name>\n'
                part_list += '</score-part>\n'
            part_list += '</part-list>\n'

            # Add the part-list to the beginning
            final_content = xml_header + doctype + score_open + part_list + cleaned_content
        else:
            # The content already has a part-list section, so just add the header
            final_content = xml_header + doctype + score_open + cleaned_content

        # Add closing tag if needed
        score_close = '</score-partwise>'
        if not final_content.strip().endswith(score_close):
            final_content += '\n' + score_close

        # Create the filename using the category, prompt ID and trial number
        if self.current_xml_filename is None:
            self.current_xml_filename = f"{self.final_output_dir}/Category{self.prompt_id.split('_')[0]}_Prompt{self.prompt_id.split('_')[1]}_Trial{self.trial_num}.xml"

        # Write to file
        with open(self.current_xml_filename, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"[Agent {self.agent_id}] Saved XML output to {self.current_xml_filename}")

        return self.current_xml_filename

    def check_for_stop(self):
        """Check if stop has been requested via Ctrl+C."""
        return self.stop_requested


class BatchMusicGenerator:
    """Class for running multiple music generation prompts in parallel"""

    def __init__(
            self,
            api_key: str,
            output_base_dir: str = r"C:\Users\Vincent\Downloads\Music\Testing",
            max_workers: int = 3,  # Number of prompts to process in parallel
            model_a_name: str = "ft:gpt-4o-mini-2024-07-18:chia:test-1-500:BH9opiWg",
            model_b_name: str = "gpt-4.1",
            model_b2_name: str = "gpt-4.1",
            model_a2_name: str = "ft:gpt-4o-mini-2024-07-18:chia:test-1-500:BH9opiWg",
            model_settings: Dict = None  # Optional settings override for temperatures, etc.
    ):
        self.api_key = api_key
        self.output_base_dir = output_base_dir
        self.max_workers = max_workers

        # Model names
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.model_b2_name = model_b2_name
        self.model_a2_name = model_a2_name

        # Model settings
        self.model_settings = {
            "model_a_temperature": 1.1,
            "model_a_top_p": 0.5,
            "model_b_temperature": 0.9,
            "model_b_top_p": 0.7,
            "model_b2_temperature": 0.9,
            "model_b2_top_p": 0.7,
            "model_a2_temperature": 0.85,
            "model_a2_top_p": 0.4,
        }

        # Override with any provided settings
        if model_settings:
            self.model_settings.update(model_settings)

        # Create output directories
        self.conversation_dir = os.path.join(output_base_dir, "Conversations")
        self.xml_output_dir = os.path.join(output_base_dir, "XML_Output")
        os.makedirs(self.conversation_dir, exist_ok=True)
        os.makedirs(self.xml_output_dir, exist_ok=True)

        # Store for active agents and their prompts
        self.running_agents = {}
        self.stop_requested = False

    def setup_signal_handler(self):
        """Set up signal handler for clean exit."""

        def signal_handler(sig, frame):
            print("\nCtrl+C detected. Cleaning up and gracefully stopping all running agents...")
            self.stop_requested = True
            for agent_id, agent in self.running_agents.items():
                agent.stop_requested = True
                print(f"Signaled agent {agent_id} to stop")
            print("Waiting for agents to complete current work...")
            time.sleep(2)  # Give agents time to save current work
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    def _process_single_prompt(self, category_id: int, prompt_id: int, prompt_text: str, trial_num: int):
        """Process a single prompt with the given trial number"""
        try:
            # Create the agent with the appropriate identifiers
            agent = LLMConversationAgent(
                api_key=self.api_key,
                model_a_name=self.model_a_name,
                model_a_temperature=self.model_settings["model_a_temperature"],
                model_a_top_p=self.model_settings["model_a_top_p"],
                model_b_name=self.model_b_name,
                model_b_temperature=self.model_settings["model_b_temperature"],
                model_b_top_p=self.model_settings["model_b_top_p"],
                model_b2_name=self.model_b2_name,
                model_b2_temperature=self.model_settings["model_b2_temperature"],
                model_b2_top_p=self.model_settings["model_b2_top_p"],
                model_a2_name=self.model_a2_name,
                model_a2_temperature=self.model_settings["model_a2_temperature"],
                model_a2_top_p=self.model_settings["model_a2_top_p"],
                conversation_dir=self.conversation_dir,
                final_output_dir=self.xml_output_dir,
                prompt_id=f"{category_id}_{prompt_id}",
                trial_num=trial_num
            )

            # Register this agent
            agent_id = f"Cat{category_id}_Prompt{prompt_id}_Trial{trial_num}"
            self.running_agents[agent_id] = agent

            # Run the conversation
            print(f"\n[Batch] Starting {agent_id}: {prompt_text[:100]}...")
            agent.generate_conversation(prompt_text, max_iterations=3, skip_initial_model_a=False)

            # Check if we got a complete score
            final_message = agent.conversation_history[-1]["content"]
            if "</score-partwise>" in final_message:
                print(f"[Batch] ✓ {agent_id}: Complete MusicXML score generated successfully")
            else:
                print(f"[Batch] ⚠ {agent_id}: Warning - XML may be incomplete (missing closing tag)")

            # Clean up
            self.running_agents.pop(agent_id, None)

            return True

        except Exception as e:
            print(f"[Batch] Error processing {category_id}_{prompt_id} trial {trial_num}: {str(e)}")
            return False

    def run_batch(self, test_prompts: Dict[int, Dict[int, str]], num_trials: int = 3):
        """
        Run a batch of prompts with multiple trials each

        Args:
            test_prompts: Dictionary of categories with prompts
                          {category_id: {prompt_id: prompt_text}}
            num_trials: Number of times to run each prompt
        """
        # Create a flat list of all work to be done
        all_tasks = []
        for category_id, prompts in test_prompts.items():
            for prompt_id, prompt_text in prompts.items():
                for trial in range(1, num_trials + 1):
                    all_tasks.append((category_id, prompt_id, prompt_text, trial))

        total_tasks = len(all_tasks)
        print(
            f"Preparing to process {total_tasks} total tasks ({len(test_prompts)} categories, {sum(len(p) for p in test_prompts.values())} prompts, {num_trials} trials each)")

        # Process tasks in parallel
        completed_tasks = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial batch of tasks
            futures = {}
            submitted_count = 0

            for idx, (category_id, prompt_id, prompt_text, trial) in enumerate(all_tasks):
                if submitted_count >= self.max_workers or idx >= len(all_tasks):
                    break

                future = executor.submit(
                    self._process_single_prompt,
                    category_id,
                    prompt_id,
                    prompt_text,
                    trial
                )

                futures[future] = (category_id, prompt_id, prompt_text, trial)
                submitted_count += 1

            # Process results and submit new tasks as others complete
            next_task_idx = submitted_count

            while futures and not self.stop_requested:
                # Wait for the next task to complete
                done, not_done = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                    timeout=30  # Check periodically even if nothing completes
                )

                # Process completed tasks
                for future in done:
                    cat_id, p_id, p_text, t_num = futures.pop(future)

                    try:
                        success = future.result()
                        task_desc = f"Category {cat_id}, Prompt {p_id}, Trial {t_num}"
                        if success:
                            print(f"[Batch] Completed {task_desc}")
                        else:
                            print(f"[Batch] Failed {task_desc}")
                    except Exception as e:
                        print(f"[Batch] Error in task (Cat{cat_id}_P{p_id}_T{t_num}): {str(e)}")

                    completed_tasks += 1
                    print(
                        f"[Batch] Progress: {completed_tasks}/{total_tasks} tasks completed ({completed_tasks / total_tasks * 100:.1f}%)")

                    # Submit next task if available
                    if next_task_idx < len(all_tasks) and not self.stop_requested:
                        next_cat_id, next_p_id, next_p_text, next_t_num = all_tasks[next_task_idx]

                        next_future = executor.submit(
                            self._process_single_prompt,
                            next_cat_id,
                            next_p_id,
                            next_p_text,
                            next_t_num
                        )

                        futures[next_future] = (next_cat_id, next_p_id, next_p_text, next_t_num)
                        next_task_idx += 1

                # Check if stop requested
                if self.stop_requested:
                    print("[Batch] Stop requested, cancelling remaining tasks...")
                    for future in list(futures.keys()):
                        future.cancel()
                    break

        print(f"[Batch] Batch processing complete. {completed_tasks}/{total_tasks} tasks finished.")
        return completed_tasks


# This is a minimal example that will run directly without command line arguments
if __name__ == "__main__":
    # Define test prompts for each category - this is minimal example with just category 1
    test_categories = {
        1: {  # Orchestration Tests
            1: "Compose a piano solo piece in C major, 24 measures",
            2: "Write a string quartet in G minor, 24 measures",
            3: "Create a brass quintet composition in F major, 24 measures",
            4: "Compose a woodwind trio piece in D minor, 24 measures",
            5: "Create a full orchestra composition in B-flat major, 24 measures"
        },
        2: {  # Specificity Tests
            1: "Write music for piano",
            2: "Compose a piano piece in C major, 24 measures",
            3: "Create a piano composition in C major with contrasting sections, 24 measures",
            4: "Write a piano piece in C major with a lyrical melody over arpeggiated accompaniment, 24 measures",
            5: "Compose a piano piece in C major, 3/4 time, beginning with a dotted-rhythm motif that transforms through development, featuring a modulation to G major in measure 12, and returning to C major by measure 20, total 24 measures"
        },
        3: {  # Length Tests
            1: "Compose a piano piece in C major with contrasting sections, 8 measures",
            2: "Compose a piano piece in C major with contrasting sections, 16 measures",
            3: "Compose a piano piece in C major with contrasting sections, 24 measures",
            4: "Compose a piano piece in C major with contrasting sections, 32 measures",
            5: "Compose a piano piece in C major with contrasting sections, 48 measures"
        },
        4: {  # Key Tests
            1: "Compose a piano piece in C major with contrasting sections, 24 measures",
            2: "Compose a piano piece in G major with contrasting sections, 24 measures",
            3: "Compose a piano piece in D major with contrasting sections, 24 measures",
            4: "Compose a piano piece in A minor with contrasting sections, 24 measures",
            5: "Compose a piano piece in E minor with contrasting sections, 24 measures"
        },
        5: {  # Time Signature Tests
            1: "Compose a piano piece in C major with contrasting sections in 4/4 time, 24 measures",
            2: "Compose a piano piece in C major with contrasting sections in 3/4 time, 24 measures",
            3: "Compose a piano piece in C major with contrasting sections in 6/8 time, 24 measures",
            4: "Compose a piano piece in C major with contrasting sections in 2/4 time, 24 measures",
            5: "Compose a piano piece in C major with contrasting sections in 9/8 time, 24 measures"
        }
    }

    # Prompt for API key
    api_key = input("Enter your OpenAI API key: ")
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)

    # Create a smaller output directory for testing
    output_dir = r"C:\Users\Vincent\Downloads\Music\Testing"

    print(f"Output will be saved to: {output_dir}")

    # Create batch generator with minimal configuration
    batch_generator = BatchMusicGenerator(
        api_key=api_key,
        output_base_dir=output_dir,
        max_workers=13,  # Just use 1 worker for testing
        model_settings={
            "model_a_temperature": 1.1,
            "model_a_top_p": 0.5,
            "model_b_temperature": 0.9,
            "model_b_top_p": 0.7,
            "model_b2_temperature": 0.9,
            "model_b2_top_p": 0.7,
            "model_a2_temperature": 0.85,
            "model_a2_top_p": 0.4,
        }
    )

    # Set up signal handler for clean exit
    batch_generator.setup_signal_handler()

    print("\nStarting batch processing with test categories...")
    batch_generator.run_batch(test_categories, num_trials=3)  # Just run 1 trial for testing

    print("\nTest complete. Check output directory for results.")