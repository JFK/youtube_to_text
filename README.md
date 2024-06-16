# youtube_to_text

`yt2t` is a command-line tool and extracts audio from YouTube video urls or a local audio file and converts it into text. It supports multiple languages and can optionally ignore YouTube transcripts.

## Features

- Extract audio from YouTube video URLs or a local audio file
- Convert audio to text using local openai-whisper library
- Support for multiple languages
- Option to ignore YouTube transcripts and download youtube audio file directly

## Installation

To install this project using pipx, use the following command:

### from PyPI
```bash
pip install yt2t
```

### from GitHub
```bash
git clone https://github.com/JFK/youtube_to_text.git
cd youtube_to_text
pipx install .
```

## Usage
To convert audio files to text, run the following command:

```bash
yt2t [YouTube video URL or local audio file path] [-l language] [-ig]
```
Example:

```bash
yt2t https://www.youtube.com/watch?v=example -l 
```

Options:

-l : Specify the language of the audio (default is ja).
-ig : Ignore YouTube direct transcripts and generate text from a provided audio file.

## License
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
