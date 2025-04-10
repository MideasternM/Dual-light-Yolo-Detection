import React, { useState, useEffect } from 'react';
import { Upload, Button, message, Card, Radio, Row, Col, Typography } from 'antd';
import { InboxOutlined, UploadOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Dragger } = Upload;
const { Text } = Typography;

const UploadArea = ({ mode, onDetectionStart, onDetectionResult }) => {
    const [fileList, setFileList] = useState([]);
    const [fileRgb, setFileRgb] = useState(null);
    const [fileIr, setFileIr] = useState(null);
    const [fileType, setFileType] = useState('image');

    useEffect(() => {
        setFileList([]);
        setFileRgb(null);
        setFileIr(null);
        if (mode === 'dual') {
            setFileType('image');
        }
    }, [mode]);

    const commonDraggerProps = (acceptType, setter) => ({
        multiple: false,
        beforeUpload: (file) => {
            let isValid = false;
            if (acceptType === 'image') {
                isValid = file.type.startsWith('image/');
                if (!isValid) message.error('Only image files are allowed!');
            } else if (acceptType === 'video') {
                isValid = file.type.startsWith('video/');
                if (!isValid) message.error('Only video files are allowed!');
            } else {
                isValid = true;
            }

            if (isValid) {
                setter(file);
            } else {
                setter(null);
            }
            return false;
        },
        onRemove: () => {
            setter(null);
        },
        maxCount: 1,
    });

    const singleProps = {
        ...commonDraggerProps(fileType, (file) => setFileList(file ? [file] : [])),
        fileList,
    };

    const rgbProps = {
        ...commonDraggerProps('image', setFileRgb),
        fileList: fileRgb ? [fileRgb] : [],
    };

    const irProps = {
        ...commonDraggerProps('image', setFileIr),
        fileList: fileIr ? [fileIr] : [],
    };

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append('mode', mode);
        let currentFileType = fileType;

        if (mode === 'single') {
            if (fileList.length === 0) {
                message.warning('Please select a file first');
                return;
            }
            formData.append('file_rgb', fileList[0]);
            formData.append('file_type', currentFileType);
        } else {
            if (!fileRgb || !fileIr) {
                message.warning('Please select both RGB and IR image files');
                return;
            }
            formData.append('file_rgb', fileRgb);
            formData.append('file_ir', fileIr);
            formData.append('file_type', 'image');
            currentFileType = 'image';
        }

        onDetectionStart();

        try {
            const response = await axios.post('/api/detect', formData, {
                headers: {
                },
                responseType: mode === 'dual' ? 'json' : 'blob',
            });

            if (mode === 'dual') {
                console.log('API Response Data (Dual Mode):', response.data);
                if (response.data && response.data.rgb_image && response.data.ir_image) {
                    const resultPayload = { rgb: response.data.rgb_image, ir: response.data.ir_image };
                    console.log('Passing to onDetectionResult:', resultPayload);
                    onDetectionResult(resultPayload, 'dual_image');
                    message.success('Dual-light detection completed!');
                } else {
                    console.error('Invalid response format received for dual mode:', response.data);
                    throw new Error('Invalid response format for dual mode');
                }
            } else {
                const url = URL.createObjectURL(response.data);
                console.log('Passing URL to onDetectionResult (Single Mode):', url, currentFileType);
                onDetectionResult(url, currentFileType);
                message.success('Single-light detection completed!');
            }

            setFileList([]);
            setFileRgb(null);
            setFileIr(null);

        } catch (error) {
            let errorMsg = 'Detection failed, please try again!';
            if (error.response) {
                try {
                    let errorDetail = null;
                    if (mode === 'dual' && error.response.data) {
                        errorDetail = error.response.data.detail;
                    } else if (error.response.data instanceof Blob) {
                        const errorJson = JSON.parse(await error.response.data.text());
                        errorDetail = errorJson.detail;
                    }
                    if (errorDetail) {
                        errorMsg = errorDetail;
                    }
                } catch (parseError) {
                    console.error('Could not parse error response', parseError);
                }
            }
            console.error('Upload failed:', error.response || error);
            message.error(errorMsg);
            onDetectionResult(null, null);
        }
    };

    const handleTypeChange = (e) => {
        setFileType(e.target.value);
        setFileList([]);
    };

    const isUploadDisabled = mode === 'single'
        ? fileList.length === 0
        : (!fileRgb || !fileIr);

    return (
        <Card title={`Upload File(s) - ${mode === 'single' ? 'Single' : 'Dual'} Mode`} style={{ marginBottom: 20 }}>
            {mode === 'single' && (
                <Radio.Group
                    onChange={handleTypeChange}
                    value={fileType}
                    style={{ marginBottom: 16 }}
                >
                    <Radio.Button value="image">Image</Radio.Button>
                    <Radio.Button value="video">Video</Radio.Button>
                </Radio.Group>
            )}

            {mode === 'single' ? (
                <Dragger {...singleProps} style={{ marginBottom: 16 }}>
                    <p className="ant-upload-drag-icon">
                        <InboxOutlined />
                    </p>
                    <p className="ant-upload-text">Click or drag {fileType === 'image' ? 'image' : 'video'} to upload</p>
                    <p className="ant-upload-hint">
                        {fileType === 'image' ? 'Supports common image formats.' : 'Supports common video formats.'}
                    </p>
                </Dragger>
            ) : (
                <Row gutter={16} style={{ marginBottom: 16 }}>
                    <Col span={12}>
                        <Text strong>RGB Image</Text>
                        <Dragger {...rgbProps}>
                            <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                            <p className="ant-upload-text">Click or drag RGB image</p>
                        </Dragger>
                    </Col>
                    <Col span={12}>
                        <Text strong>IR Image</Text>
                        <Dragger {...irProps}>
                            <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                            <p className="ant-upload-text">Click or drag IR image</p>
                        </Dragger>
                    </Col>
                </Row>
            )}

            <Button
                type="primary"
                onClick={handleUpload}
                disabled={isUploadDisabled}
                style={{ marginTop: 16 }}
                icon={<UploadOutlined />}
            >
                Start Detection
            </Button>
        </Card>
    );
};

export default UploadArea;