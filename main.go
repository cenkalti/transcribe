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
	openAIAPIURL = "https://api.openai.com/v1/audio/transcriptions"
	model        = "gpt-4o-transcribe-diarize"
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

	// Transcribe with diarization
	fmt.Println("Transcribing audio with speaker diarization...")
	transcription, err := transcribeAudio(mp3File, os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		fmt.Printf("Error transcribing audio: %v\n", err)
		os.Exit(1)
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
	if err := writer.WriteField("response_format", "verbose_json"); err != nil {
		return nil, fmt.Errorf("failed to write response_format field: %w", err)
	}

	// Add timestamp_granularities field for segment-level timestamps
	if err := writer.WriteField("timestamp_granularities[]", "segment"); err != nil {
		return nil, fmt.Errorf("failed to write timestamp_granularities field: %w", err)
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
