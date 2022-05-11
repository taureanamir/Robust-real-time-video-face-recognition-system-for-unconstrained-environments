from ConfigManager import ConfigManager

cfg = ConfigManager.Instance()

camnum = 2

cameraConfigDict = {}

for i in range(1, cfg.iNumCameras + 1):
    setattr(None, camCodec, "cfg.cam" + str(i) + "_codec")
    camCodec = "cfg.cam" + str(i) + "_codec"
    camNum = "cfg.cam" + str(i) + "_sCamNum"
    camImgHeight = "cfg.cam" + str(i) + "_iImgHeight"
    camImgWidth = "cfg.cam" + str(i) + "_iImgWidth"
    camName = "cfg.cam" + str(i) + "_sName"
    camVideoPath = "cfg.cam" + str(i) + "_sVideoPath"
    camHasLock = "cfg.cam" + str(i) + "_hasLock"

    addCam = {i: [camCodec, camNum, camImgHeight,
                  camImgWidth, camName, camVideoPath, camHasLock]}

    print("=---------------- add caom", addCam)

    cameraConfigDict.update(addCam)


cameraConfigDict = {
    1: [cfg.cam1_codec, cfg.cam1_sCamNum, cfg.cam1_iImgHeight, cfg.cam1_iImgWidth, cfg.cam1_sName, cfg.cam1_sVideoPath, cfg.cam1_hasLock],
    2: [cfg.cam2_codec, cfg.cam2_sCamNum, cfg.cam2_iImgHeight, cfg.cam2_iImgWidth, cfg.cam2_sName, cfg.cam2_sVideoPath, cfg.cam2_hasLock],
    3: [cfg.cam3_codec, cfg.cam3_sCamNum, cfg.cam3_iImgHeight, cfg.cam3_iImgWidth, cfg.cam3_sName, cfg.cam3_sVideoPath, cfg.cam3_hasLock],
    4: [cfg.cam4_codec, cfg.cam4_sCamNum, cfg.cam4_iImgHeight, cfg.cam4_iImgWidth, cfg.cam4_sName, cfg.cam4_sVideoPath, cfg.cam4_hasLock]
}

for i in range(1, cfg.iNumCameras + 1):
    camCodec = ''
    setattr(None, 'camCodec', "cfg.cam" + str(i) + "_codec")

cameraConfig = cameraConfigDict[camnum]

print(cameraConfig)
print("--------------------------")

camera_id = cameraConfig[1]
camera_name = cameraConfig[4]
camera_stream = cameraConfig[5]
hasLock = cameraConfig[6]

print("camera_id: ", camera_id)
print("camera_name: ", camera_name)
print("camera_stream: ", camera_stream)
print("hasLock: ", hasLock)

print("--------------------------")

for i in range(0, 7):
    print(cameraConfig[i])
