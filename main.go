package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

const (
	assemblyAIBaseURL = "https://api.assemblyai.com/v2"
)

// Utterance represents a single transcribed utterance with speaker info
type Utterance struct {
	Speaker    string  `json:"speaker"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
}

// TranscriptRequest represents the request to create a transcript
type TranscriptRequest struct {
	AudioURL      string `json:"audio_url"`
	SpeakerLabels bool   `json:"speaker_labels"`
}

// TranscriptionResponse represents the API response
type TranscriptionResponse struct {
	ID         string      `json:"id"`
	Status     string      `json:"status"`
	Text       string      `json:"text"`
	Utterances []Utterance `json:"utterances"`
	Error      string      `json:"error"`
}

// UploadResponse represents the upload endpoint response
type UploadResponse struct {
	UploadURL string `json:"upload_url"`
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

	apiKey := os.Getenv("ASSEMBLYAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: ASSEMBLYAI_API_KEY not found in .env")
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

	// Upload audio file
	fmt.Println("Uploading audio file...")
	uploadURL, err := uploadAudio(mp3File, apiKey)
	if err != nil {
		fmt.Printf("Error uploading audio: %v\n", err)
		os.Exit(1)
	}

	// Transcribe with diarization
	fmt.Println("Transcribing audio with speaker diarization...")
	transcription, err := transcribeAudio(uploadURL, apiKey)
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

// uploadAudio uploads an audio file to AssemblyAI and returns the upload URL
func uploadAudio(audioFile, apiKey string) (string, error) {
	file, err := os.Open(audioFile)
	if err != nil {
		return "", fmt.Errorf("failed to open audio file: %w", err)
	}
	defer file.Close()

	req, err := http.NewRequest("POST", assemblyAIBaseURL+"/upload", file)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", apiKey)
	req.Header.Set("Content-Type", "application/octet-stream")

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("upload failed with status %d: %s", resp.StatusCode, string(body))
	}

	var uploadResp UploadResponse
	if err := json.Unmarshal(body, &uploadResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	return uploadResp.UploadURL, nil
}

// transcribeAudio submits audio for transcription and polls until complete
func transcribeAudio(audioURL, apiKey string) (*TranscriptionResponse, error) {
	// Submit transcription request
	requestData := TranscriptRequest{
		AudioURL:      audioURL,
		SpeakerLabels: true,
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", assemblyAIBaseURL+"/transcript", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("transcription request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var transcription TranscriptionResponse
	if err := json.Unmarshal(body, &transcription); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	transcriptID := transcription.ID

	// Poll for completion
	pollingURL := fmt.Sprintf("%s/transcript/%s", assemblyAIBaseURL, transcriptID)

	for {
		req, err := http.NewRequest("GET", pollingURL, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create polling request: %w", err)
		}

		req.Header.Set("Authorization", apiKey)

		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to poll: %w", err)
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to read polling response: %w", err)
		}

		if err := json.Unmarshal(body, &transcription); err != nil {
			return nil, fmt.Errorf("failed to parse polling response: %w", err)
		}

		switch transcription.Status {
		case "completed":
			return &transcription, nil
		case "error":
			return nil, fmt.Errorf("transcription failed: %s", transcription.Error)
		case "queued", "processing":
			fmt.Printf("Status: %s... waiting\n", transcription.Status)
			time.Sleep(3 * time.Second)
		default:
			return nil, fmt.Errorf("unexpected status: %s", transcription.Status)
		}
	}
}

// saveTranscription saves the transcription to a text file with speaker labels and timestamps
func saveTranscription(filename string, transcription *TranscriptionResponse) error {
	var output strings.Builder

	// If we have utterances with speaker info, format them nicely
	if len(transcription.Utterances) > 0 {
		currentSpeaker := ""
		for _, utterance := range transcription.Utterances {
			// Format timestamps (convert milliseconds to HH:MM:SS)
			startTime := formatTimestamp(float64(utterance.Start) / 1000.0)
			endTime := formatTimestamp(float64(utterance.End) / 1000.0)

			speaker := utterance.Speaker
			if speaker == "" {
				speaker = "Unknown"
			}

			// Add speaker header if speaker changes
			if speaker != currentSpeaker {
				if currentSpeaker != "" {
					output.WriteString("\n")
				}
				output.WriteString(fmt.Sprintf("[%s - %s] Speaker %s:\n", startTime, endTime, speaker))
				currentSpeaker = speaker
			} else {
				output.WriteString(fmt.Sprintf("[%s - %s] ", startTime, endTime))
			}

			output.WriteString(strings.TrimSpace(utterance.Text))
			output.WriteString("\n")
		}
	} else {
		// Fallback to plain text if no utterances
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
