import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
  Dimensions,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor, runAtTargetFps } from 'react-native-vision-camera';
import { useNavigation, useIsFocused } from '@react-navigation/native';
import { Linking } from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useTextRecognition, PhotoRecognizer } from 'react-native-vision-camera-text-recognition';
import { runOnJS } from 'react-native-reanimated';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

const ScanLicensePlateScreen = () => {
  const navigation = useNavigation();
  const isFocused = useIsFocused();
  const camera = useRef(null);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const [isProcessing, setIsProcessing] = useState(false);

  // Load YOLOv9 TFLite model
  const objectDetection = useTensorflowModel(require('../assets/anpr2_yolov9_int8.tflite'));
  const model = objectDetection.state === 'loaded' ? objectDetection.model : null;

  // Initialize resize plugin
  const { resize } = useResizePlugin();

  // Initialize text recognition
  const { scanText } = useTextRecognition({ language: 'latin' });

  useEffect(() => {
    if (!hasPermission) {
      requestPermission().then((granted) => {
        if (!granted) {
          Alert.alert(
            'Camera Permission Required',
            'Please enable camera access in settings to use this feature.',
            [
              { text: 'Cancel', style: 'cancel' },
              { text: 'Open Settings', onPress: () => Linking.openSettings() },
            ]
          );
        }
      });
    }
  }, [hasPermission]);

  const processLicensePlate = (licensePlateText) => {
    if (licensePlateText) {
      const cleanedText = licensePlateText.trim().replace(/\s+/g, '');
      navigation.navigate('AddCar', { licensePlate: cleanedText });
    } else {
      Alert.alert('No Text Found', 'Could not detect a license plate. Please try again.');
    }
    setIsProcessing(false);
  };

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      runAtTargetFps(2, () => {
        'worklet';
        if (model == null || isProcessing) return;

        // Resize frame to 640x640 for YOLOv9 model
        const resized = resize(frame, {
          scale: {
            width: 640,
            height: 640,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        // Convert uint8 to float32
        const inputArray = new Float32Array(resized.length);
        for (let i = 0; i < resized.length; i++) {
          inputArray[i] = resized[i] / 255.0; // Normalize to [0,1]
        }

        // Run YOLOv9 model
        const outputs = model.runSync([inputArray]);

        // Process outputs: [1, 5, 8400] -> [x, y, w, h, confidence] for each detection
        const output = outputs[0]; // Shape: [1, 5, 8400]
        let maxConfidence = 0;
        let bestBox = null;

        for (let i = 0; i < 8400; i++) {
          const confidence = output[4 * 8400 + i]; // Confidence score
          if (confidence > maxConfidence && confidence > 0.5) {
            maxConfidence = confidence;
            const x = output[0 * 8400 + i]; // Center x
            const y = output[1 * 8400 + i]; // Center y
            const w = output[2 * 8400 + i]; // Width
            const h = output[3 * 8400 + i]; // Height
            bestBox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]; // Convert to [x1, y1, x2, y2]
          }
        }

        if (bestBox) {
          const [x1, y1, x2, y2] = bestBox;

          // Calculate crop coordinates (scale back to original frame dimensions)
          const frameWidth = frame.width;
          const frameHeight = frame.height;
          const scaleX = frameWidth / 640;
          const scaleY = frameHeight / 640;

          const cropX = Math.max(0, x1 * scaleX);
          const cropY = Math.max(0, y1 * scaleY);
          const cropWidth = Math.min(frameWidth - cropX, (x2 - x1) * scaleX);
          const cropHeight = Math.min(frameHeight - cropY, (y2 - y1) * scaleY);

          // Perform OCR on full frame and filter text within bounding box
          const result = scanText(frame);
          let detectedText = '';
          for (const block of result.result.blocks) {
            const { x, y, width, height } = block.frame;
            // Check if text block is within the detected bounding box
            if (
              x >= cropX &&
              y >= cropY &&
              x + width <= cropX + cropWidth &&
              y + height <= cropY + cropHeight
            ) {
              detectedText += block.text + ' ';
            }
          }

          // Alternative: Use vision-camera-cropper to crop and OCR (uncomment if installed)
          /*
          const croppedPath = crop(frame, {
            cropRegion: {
              left: cropX,
              top: cropY,
              width: cropWidth,
              height: cropHeight,
            },
          });
          const ocrResult = runOnJS(PhotoRecognizer)({ uri: croppedPath, orientation: 'portrait' });
          runOnJS(processLicensePlate)(ocrResult.text);
          */

          runOnJS(processLicensePlate)(detectedText);
        }
      });
    },
    [model, isProcessing]
  );

  const capturePhoto = async () => {
    if (camera.current && !isProcessing) {
      setIsProcessing(true);
      try {
        // Capture photo with explicit options
        const photo = await camera.current.takePhoto({
          flash: 'off',
          enableShutterSound: false,
        });
        console.log('Photo captured:', photo);

        // Ensure the path is valid
        const imagePath = photo.path.startsWith('file://') ? photo.path : `file://${photo.path}`;
        console.log('Image path for OCR:', imagePath);

        // Perform OCR
        const result = await PhotoRecognizer({ uri: imagePath, orientation: 'portrait' });
        console.log('OCR result:', result);
        processLicensePlate(result.text);
      } catch (error) {
        console.error('Error capturing or processing photo:', error);
        Alert.alert('Error', `Failed to process the photo: ${error.message}. Please try again.`);
        setIsProcessing(false);
      }
    }
  };

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>
          No camera device available. Please ensure your device has a back camera or try again.
        </Text>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.retryText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isFocused}
        photo={true}
        frameProcessor={frameProcessor}
        onInitialized={() => console.log('Camera initialized')}
        onError={(error) => {
          console.error('Camera error:', error);
          Alert.alert('Camera Error', error.message);
        }}
      />

      <TouchableOpacity
        style={styles.closeButton}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.closeText}>✕</Text>
      </TouchableOpacity>

      <View style={styles.bottomContainer}>
        <View style={styles.textWrapper}>
          <Text style={styles.bottomTitle}>Scan License Plate</Text>
          <Text style={styles.bottomSubtitle}>Position License Plate in frame</Text>
        </View>

        <TouchableOpacity
          style={[styles.captureButton, isProcessing && styles.disabledButton]}
          onPress={capturePhoto}
          disabled={isProcessing}
        >
          <Text style={styles.captureText}>
            {isProcessing ? 'Processing...' : 'Capture'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.manualEntry}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.manualText}>Enter License Plate Manually</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingText: {
    color: '#fff',
    marginTop: 12,
    fontSize: 16,
    textAlign: 'center',
  },
  errorText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginHorizontal: 20,
  },
  retryButton: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#3b82f6',
    borderRadius: 5,
  },
  retryText: {
    color: '#fff',
    fontSize: 16,
  },
  closeButton: {
    position: 'absolute',
    top: 80,
    left: 20,
    zIndex: 10,
  },
  closeText: {
    fontSize: 28,
    color: '#fff',
  },
  bottomContainer: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    height: SCREEN_HEIGHT * 0.4,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'flex-start',
    paddingTop: 30,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  textWrapper: {
    alignItems: 'center',
    marginBottom: 20,
  },
  bottomTitle: {
    color: '#fff',
    fontSize: 40,
    fontWeight: 'bold',
    marginBottom: 6,
  },
  bottomSubtitle: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
  },
  captureButton: {
    backgroundColor: '#3b82f6',
    paddingVertical: 12,
    paddingHorizontal: 40,
    borderRadius: 8,
    marginBottom: 20,
  },
  disabledButton: {
    backgroundColor: '#666',
  },
  captureText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  manualEntry: {
    position: 'absolute',
    bottom: 25,
  },
  manualText: {
    color: '#3b82f6',
    fontSize: 16,
    textDecorationLine: 'underline',
  },
});

export default ScanLicensePlateScreen;