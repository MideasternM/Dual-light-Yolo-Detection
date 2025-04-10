import React, { useState } from 'react';
import { Layout, Typography, theme, Radio, Space } from 'antd';
import UploadArea from './components/UploadArea';
import ResultDisplay from './components/ResultDisplay';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

const App = () => {
    const [resultData, setResultData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [fileType, setFileType] = useState(null);
    const [mode, setMode] = useState('single');

    const { token } = theme.useToken();

    const handleDetectionResult = (result, type) => {
        setResultData(result);
        setFileType(type);
        setLoading(false);
    };

    const handleModeChange = (e) => {
        setMode(e.target.value);
        setResultData(null);
        setFileType(null);
    };

    return (
        <Layout className="layout">
            <Header style={{ background: token.colorBgContainer }}>
                <div className="logo">
                    <Title level={3} style={{ margin: '16px 0' }}>YOLO Object Detection System</Title>
                </div>
            </Header>
            <Content style={{ padding: '0 50px' }}>
                <div className="site-layout-content" style={{ margin: '16px 0', padding: 24, background: token.colorBgContainer }}>
                    <Space direction="vertical" size="large" style={{ display: 'flex', marginBottom: 24 }}>
                        <Radio.Group onChange={handleModeChange} value={mode}>
                            <Radio value={'single'}>Single-Light Mode</Radio>
                            <Radio value={'dual'}>Dual-Light Mode (Image Only)</Radio>
                        </Radio.Group>
                    </Space>
                    <UploadArea
                        mode={mode}
                        onDetectionStart={() => setLoading(true)}
                        onDetectionResult={handleDetectionResult}
                    />
                    <ResultDisplay
                        resultData={resultData}
                        loading={loading}
                        fileType={fileType}
                    />
                </div>
            </Content>
            <Footer style={{ textAlign: 'center' }}>
                YOLO Object Detection System ©{new Date().getFullYear()} Created with React & FastAPI
            </Footer>
        </Layout>
    );
};

export default App;