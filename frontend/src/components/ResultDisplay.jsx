import React from 'react';
import { Card, Spin, Empty, Image, Row, Col, Typography } from 'antd';

const { Text } = Typography;

const ResultDisplay = ({ resultData, loading, fileType }) => {

    const renderContent = () => {
        // Log props received by ResultDisplay
        console.log('ResultDisplay Props:', { resultData, loading, fileType });

        if (loading) {
            return (
                <div style={{ textAlign: 'center', padding: '40px 0' }}>
                    <Spin size="large" tip="Object detection in progress..." />
                </div>
            );
        }

        if (!resultData) {
            return <Empty description="No detection results yet" />;
        }

        // Handle different file types
        switch (fileType) {
            case 'image':
                console.log('Rendering single image with URL:', resultData);
                // resultData is a URL string
                return (
                    <div style={{ textAlign: 'center' }}>
                        <Image
                            src={resultData}
                            alt="Detection Result"
                            style={{ maxWidth: '100%', maxHeight: '70vh' }}
                        />
                    </div>
                );
            case 'video':
                console.log('Rendering video with URL:', resultData);
                // resultData is a URL string (object URL)
                return (
                    <div style={{ textAlign: 'center' }}>
                        <video
                            controls
                            style={{ maxWidth: '100%', maxHeight: '70vh' }}
                            src={resultData}
                        >
                            Your browser does not support the video tag.
                        </video>
                    </div>
                );
            case 'dual_image':
                // Log the data received for dual image rendering
                console.log('Rendering dual images with data:', resultData);
                // resultData is an object { rgb: base64, ir: base64 }
                const rgbSrc = `data:image/jpeg;base64,${resultData.rgb}`;
                const irSrc = `data:image/jpeg;base64,${resultData.ir}`;
                // Log the generated sources
                console.log('Generated RGB Src:', rgbSrc.substring(0, 100) + '...'); // Log prefix
                console.log('Generated IR Src:', irSrc.substring(0, 100) + '...'); // Log prefix
                return (
                    <Row gutter={16}>
                        <Col span={12} style={{ textAlign: 'center' }}>
                            <Text strong>Processed RGB Image</Text>
                            <Image
                                src={rgbSrc}
                                alt="Processed RGB Result"
                                style={{ maxWidth: '100%', maxHeight: '70vh' }}
                            />
                        </Col>
                        <Col span={12} style={{ textAlign: 'center' }}>
                            <Text strong>Processed IR Image</Text>
                            <Image
                                src={irSrc}
                                alt="Processed IR Result"
                                style={{ maxWidth: '100%', maxHeight: '70vh' }}
                            />
                        </Col>
                    </Row>
                );
            default:
                console.log('Rendering default Empty state');
                // Should not happen, but handle gracefully
                return <Empty description="Unknown result type" />;
        }
    };

    return (
        <Card title="Detection Result" style={{ marginTop: 20 }}>
            {renderContent()}
        </Card>
    );
};

export default ResultDisplay;