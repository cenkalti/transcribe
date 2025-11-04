package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

const (
	openAIAPIURL    = "https://api.openai.com/v1/audio/transcriptions"
	model           = "gpt-4o-transcribe-diarize"
	maxDuration     = 1400 // Maximum duration in seconds for the diarization model
	chunkDuration   = 1200 // Split into 20-minute chunks to stay under the limit
)

// DiarizedSegment represents a single transcribed segment with speaker info
type DiarizedSegment struct {
	Speaker string  `json:"speaker"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Text    string  `json:"text"`
}

// TranscriptionResponse represents the API response
type TranscriptionResponse struct {
	Text     string            `json:"text"`
	Segments []DiarizedSegment `json:"segments"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: transcribe <video-file>")
		os.Exit(1)
	}

	videoFile := os.Args[1]

	// Load API key from .env
	err := godotenv.Load()
	if err != nil {
		fmt.Printf("Error loading .env file: %v\n", err)
		os.Exit(1)
	}

	// Convert video to MP3
	fmt.Println("Converting video to MP3...")
	mp3File, err := convertToMP3(videoFile)
	if err != nil {
		fmt.Printf("Error converting video: %v\n", err)
		os.Exit(1)
	}
	defer os.Remove(mp3File)

	// Get audio duration
	duration, err := getAudioDuration(mp3File)
	if err != nil {
		fmt.Printf("Error getting audio duration: %v\n", err)
		os.Exit(1)
	}

	apiKey := os.Getenv("OPENAI_API_KEY")

	// Transcribe with diarization
	var transcription *TranscriptionResponse
	if duration > maxDuration {
		// Split into chunks
		fmt.Printf("Audio is %.0f seconds, splitting into chunks...\n", duration)
		transcription, err = transcribeAudioInChunks(mp3File, apiKey, duration)
		if err != nil {
			fmt.Printf("Error transcribing audio: %v\n", err)
			os.Exit(1)
		}
	} else {
		fmt.Println("Transcribing audio with speaker diarization...")
		transcription, err = transcribeAudio(mp3File, apiKey)
		if err != nil {
			fmt.Printf("Error transcribing audio: %v\n", err)
			os.Exit(1)
		}
	}

	// Save to output file
	outputFile := strings.TrimSuffix(videoFile, filepath.Ext(videoFile)) + ".txt"
	err = saveTranscription(outputFile, transcription)
	if err != nil {
		fmt.Printf("Error saving transcription: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Transcription saved to: %s\n", outputFile)
}

// getAudioDuration returns the duration of an audio file in seconds using FFmpeg
func getAudioDuration(audioFile string) (float64, error) {
	cmd := exec.Command("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audioFile)
	var out bytes.Buffer
	cmd.Stdout = &out

	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf("ffprobe failed: %w", err)
	}

	var duration float64
	_, err := fmt.Sscanf(strings.TrimSpace(out.String()), "%f", &duration)
	if err != nil {
		return 0, fmt.Errorf("failed to parse duration: %w", err)
	}

	return duration, nil
}

// splitAudioIntoChunks splits an audio file into chunks of specified duration using FFmpeg
func splitAudioIntoChunks(audioFile string, chunkDuration int) ([]string, error) {
	duration, err := getAudioDuration(audioFile)
	if err != nil {
		return nil, err
	}

	var chunks []string
	numChunks := int(duration)/chunkDuration + 1

	for i := 0; i < numChunks; i++ {
		startTime := i * chunkDuration

		// Create temporary chunk file
		tmpFile, err := os.CreateTemp("", fmt.Sprintf("chunk-%d-*.mp3", i))
		if err != nil {
			// Clean up previously created chunks
			for _, chunk := range chunks {
				os.Remove(chunk)
			}
			return nil, fmt.Errorf("failed to create temp chunk file: %w", err)
		}
		tmpFile.Close()
		chunkPath := tmpFile.Name()

		// Extract chunk using FFmpeg
		cmd := exec.Command("ffmpeg", "-i", audioFile, "-ss", fmt.Sprintf("%d", startTime), "-t", fmt.Sprintf("%d", chunkDuration), "-acodec", "copy", chunkPath, "-y")
		var stderr bytes.Buffer
		cmd.Stderr = &stderr

		if err := cmd.Run(); err != nil {
			// Clean up
			os.Remove(chunkPath)
			for _, chunk := range chunks {
				os.Remove(chunk)
			}
			return nil, fmt.Errorf("ffmpeg chunk extraction failed: %w\nOutput: %s", err, stderr.String())
		}

		chunks = append(chunks, chunkPath)
	}

	return chunks, nil
}

// transcribeAudioInChunks splits audio and transcribes each chunk, combining results
func transcribeAudioInChunks(audioFile, apiKey string, duration float64) (*TranscriptionResponse, error) {
	// Split audio into chunks
	chunks, err := splitAudioIntoChunks(audioFile, chunkDuration)
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, chunk := range chunks {
			os.Remove(chunk)
		}
	}()

	fmt.Printf("Split into %d chunks\n", len(chunks))

	// Transcribe each chunk
	var allSegments []DiarizedSegment
	var fullText strings.Builder
	var timeOffset float64

	for i, chunk := range chunks {
		fmt.Printf("Transcribing chunk %d/%d...\n", i+1, len(chunks))

		transcription, err := transcribeAudio(chunk, apiKey)
		if err != nil {
			return nil, fmt.Errorf("failed to transcribe chunk %d: %w", i, err)
		}

		// Adjust timestamps and append segments
		for _, segment := range transcription.Segments {
			segment.Start += timeOffset
			segment.End += timeOffset
			allSegments = append(allSegments, segment)
		}

		if transcription.Text != "" {
			fullText.WriteString(transcription.Text)
			fullText.WriteString(" ")
		}

		timeOffset += float64(chunkDuration)
	}

	return &TranscriptionResponse{
		Text:     strings.TrimSpace(fullText.String()),
		Segments: allSegments,
	}, nil
}

// convertToMP3 converts a video file to MP3 format using FFmpeg
func convertToMP3(videoFile string) (string, error) {
	// Check if input file exists
	if _, err := os.Stat(videoFile); os.IsNotExist(err) {
		return "", fmt.Errorf("video file does not exist: %s", videoFile)
	}

	// Create temporary MP3 file
	tmpFile, err := os.CreateTemp("", "transcribe-*.mp3")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpFile.Close()

	mp3Path := tmpFile.Name()

	// Run FFmpeg to convert video to MP3
	cmd := exec.Command("ffmpeg", "-i", videoFile, "-vn", "-acodec", "libmp3lame", "-q:a", "2", mp3Path, "-y")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		os.Remove(mp3Path)
		return "", fmt.Errorf("ffmpeg failed: %w\nOutput: %s", err, stderr.String())
	}

	return mp3Path, nil
}

// transcribeAudio sends the audio file to OpenAI API for transcription with diarization
func transcribeAudio(audioFile, apiKey string) (*TranscriptionResponse, error) {
	// Open the audio file
	file, err := os.Open(audioFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open audio file: %w", err)
	}
	defer file.Close()

	// Create multipart form
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// Add file field
	part, err := writer.CreateFormFile("file", filepath.Base(audioFile))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		return nil, fmt.Errorf("failed to copy file: %w", err)
	}

	// Add model field
	if err := writer.WriteField("model", model); err != nil {
		return nil, fmt.Errorf("failed to write model field: %w", err)
	}

	// Add response_format field
	if err := writer.WriteField("response_format", "diarized_json"); err != nil {
		return nil, fmt.Errorf("failed to write response_format field: %w", err)
	}

	// Add chunking_strategy field
	if err := writer.WriteField("chunking_strategy", "auto"); err != nil {
		return nil, fmt.Errorf("failed to write chunking_strategy field: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequest("POST", openAIAPIURL, &requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var transcription TranscriptionResponse
	if err := json.Unmarshal(body, &transcription); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &transcription, nil
}

// saveTranscription saves the transcription to a text file with speaker labels and timestamps
func saveTranscription(filename string, transcription *TranscriptionResponse) error {
	var output strings.Builder

	// If we have segments with speaker info, format them nicely
	if len(transcription.Segments) > 0 {
		currentSpeaker := ""
		for _, segment := range transcription.Segments {
			// Format timestamps
			startTime := formatTimestamp(segment.Start)
			endTime := formatTimestamp(segment.End)

			speaker := segment.Speaker
			if speaker == "" {
				speaker = "Unknown"
			}

			// Add speaker header if speaker changes
			if speaker != currentSpeaker {
				if currentSpeaker != "" {
					output.WriteString("\n")
				}
				output.WriteString(fmt.Sprintf("[%s - %s] %s:\n", startTime, endTime, speaker))
				currentSpeaker = speaker
			} else {
				output.WriteString(fmt.Sprintf("[%s - %s] ", startTime, endTime))
			}

			output.WriteString(strings.TrimSpace(segment.Text))
			output.WriteString("\n")
		}
	} else {
		// Fallback to plain text if no segments
		output.WriteString(transcription.Text)
		output.WriteString("\n")
	}

	return os.WriteFile(filename, []byte(output.String()), 0644)
}

// formatTimestamp converts seconds to HH:MM:SS format
func formatTimestamp(seconds float64) string {
	duration := time.Duration(seconds * float64(time.Second))
	hours := int(duration.Hours())
	minutes := int(duration.Minutes()) % 60
	secs := int(duration.Seconds()) % 60
	return fmt.Sprintf("%02d:%02d:%02d", hours, minutes, secs)
}
