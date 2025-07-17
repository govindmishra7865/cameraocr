import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Alert, Dimensions, Linking } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { useNavigation, useIsFocused } from '@react-navigation/native';
import Tflite from 'react-native-tflite';
import TextRecognition from '@react-native-ml-kit/text-recognition';
import ImageEditor from "@react-native-community/image-editor";

const { height: SCREEN_HEIGHT } = Dimensions.get('window');
const tflite = new Tflite();

const ScanLicensePlateScreen = () => {
  const navigation = useNavigation();
  const isFocused = useIsFocused();
  const camera = useRef(null);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
    tflite.loadModel({
      model: 'anpr2_yolov9_int8.tflite',
      labels: '', // not needed for object detection
    }, (err) => { if (err) console.error('TFLite load error', err); });
  }, [hasPermission]);

  const cropImageToBBox = async (uri, bbox) => {
    const cropData = {
      offset: { x: bbox.x * bbox.imageWidth, y: bbox.y * bbox.imageHeight },
      size: { width: bbox.w * bbox.imageWidth, height: bbox.h * bbox.imageHeight },
    };
    return await ImageEditor.cropImage(uri, cropData);
  };

  const capturePhoto = async () => {
    if (!camera.current || isProcessing) return;
    setIsProcessing(true);
    try {
      const photo = await camera.current.takePhoto({ skipMetadata: true });
      const uri = `file://${photo.path}`;

      //  Run object detection
      tflite.detectObjectOnImage({
        path: uri,
        threshold: 0.5,
        numResults: 1,
      }, async (err, res) => {
        if (err || !res?.length) {
          Alert.alert('Detection Failed', 'Could not detect a number plate.');
          setIsProcessing(false);
          return;
        }

        const obj = res[0];
        const { rect, imageWidth, imageHeight } = obj;
        const cropUri = await cropImageToBBox(uri, { ...rect, imageWidth, imageHeight });

        //  OCR on cropped image
        const ocrResult = await TextRecognition.recognize(cropUri);
        const text = ocrResult.text?.replace(/\s+/g, '');

        if (text) navigation.navigate('AddCar', { licensePlate: text });
        else Alert.alert('OCR Failed', 'Could not read plate text.');
        setIsProcessing(false);
      });
    } catch (e) {
      console.error(e);
      Alert.alert('Error', 'Something went wrong.');
      setIsProcessing(false);
    }
  };

  if (!hasPermission) return <View style={styles.container}><ActivityIndicator /><Text style={styles.loadingText}>Camera permission needed</Text></View>;
  if (!device) return <View style={styles.container}><Text style={styles.loadingText}>No camera found</Text></View>;

  return (
    <View style={styles.container}>
      <Camera ref={camera} style={StyleSheet.absoluteFill} device={device} isActive={isFocused} photo={true} />
      <TouchableOpacity onPress={() => navigation.goBack()} style={styles.closeButton}><Text style={styles.closeText}>✕</Text></TouchableOpacity>
      <View style={styles.bottomContainer}>
        <TouchableOpacity disabled={isProcessing} onPress={capturePhoto} style={[styles.captureButton, isProcessing && styles.disabled]} >
          <Text style={styles.captureText}>{isProcessing ? 'Processing...' : 'Capture'}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  loadingText: { color: '#fff', marginTop: 20, textAlign: 'center' },
  closeButton: { position: 'absolute', top: 40, left: 20, zIndex: 10 },
  closeText: { fontSize: 28, color: '#fff' },
  bottomContainer: { position: 'absolute', bottom: 50, width: '100%', alignItems: 'center' },
  captureButton: { backgroundColor: '#3b82f6', padding: 20, borderRadius: 40 },
  disabled: { backgroundColor: '#555' },
  captureText: { fontSize: 16, color: '#fff', fontWeight: '600' },
});

export default ScanLicensePlateScreen;
