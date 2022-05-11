import confuse
import os
from pathlib import Path


class Singleton:

    def __init__(self, cls):
        self._cls = cls

    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)


@Singleton
class ConfigManager():
    def __init__(self):
        # Application root level directory
        self.APPDIR = os.path.normpath(
            str(Path(__file__).resolve().parents[2]))
        # print(self.APPDIR)
        os.environ["FACE_RECOGNITION_TO_ACCESS_ELECTRONIC_DOOR_SYSTEMDIR"] = self.APPDIR
        self.config_file = confuse.Configuration(
            'face_recognition_to_access_electronic_door_system')
        self.config_file_path = os.path.join(
            self.APPDIR, 'input', 'config', 'config.yaml')
        self.config_file.set_file(self.config_file_path)
        config_filename = os.path.join(
            self.config_file.config_dir(), confuse.CONFIG_FILENAME)

        # camera configs
        self.iNumCameras = self.config_file['iNumCameras'].get(int)
        # cam1
        self.cam1_codec = self.config_file['cam1']['codec'].get(str)
        self.cam1_sCamNum = self.config_file['cam1']['iCamNum'].get(str)
        self.cam1_iImgHeight = self.config_file['cam1']['iImgHeight'].get(int)
        self.cam1_iImgWidth = self.config_file['cam1']['iImgWidth'].get(int)
        self.cam1_sName = self.config_file['cam1']['sName'].get(str)
        self.cam1_sVideoPath = self.config_file['cam1']['sVideoPath'].get(str)
        self.cam1_sNodeJSPostUrl = self.config_file['cam1']['sNodeJSPostUrl'].get(
            str)

        # cam2
        self.cam2_codec = self.config_file['cam2']['codec'].get(str)
        self.cam2_sCamNum = self.config_file['cam2']['iCamNum'].get(str)
        self.cam2_iImgHeight = self.config_file['cam2']['iImgHeight'].get(int)
        self.cam2_iImgWidth = self.config_file['cam2']['iImgWidth'].get(int)
        self.cam2_sName = self.config_file['cam2']['sName'].get(str)
        self.cam2_sVideoPath = self.config_file['cam2']['sVideoPath'].get(str)
        self.cam2_sNodeJSPostUrl = self.config_file['cam2']['sNodeJSPostUrl'].get(
            str)

        # cam3
        self.cam3_codec = self.config_file['cam3']['codec'].get(str)
        self.cam3_sCamNum = self.config_file['cam3']['iCamNum'].get(str)
        self.cam3_iImgHeight = self.config_file['cam3']['iImgHeight'].get(int)
        self.cam3_iImgWidth = self.config_file['cam3']['iImgWidth'].get(int)
        self.cam3_sName = self.config_file['cam3']['sName'].get(str)
        self.cam3_sVideoPath = self.config_file['cam3']['sVideoPath'].get(str)
        self.cam3_sNodeJSPostUrl = self.config_file['cam3']['sNodeJSPostUrl'].get(
            str)

        # cam4
        self.cam4_codec = self.config_file['cam4']['codec'].get(str)
        self.cam4_sCamNum = self.config_file['cam4']['iCamNum'].get(str)
        self.cam4_iImgHeight = self.config_file['cam4']['iImgHeight'].get(int)
        self.cam4_iImgWidth = self.config_file['cam4']['iImgWidth'].get(int)
        self.cam4_sName = self.config_file['cam4']['sName'].get(str)
        self.cam4_sVideoPath = self.config_file['cam4']['sVideoPath'].get(str)
        self.cam4_sNodeJSPostUrl = self.config_file['cam4']['sNodeJSPostUrl'].get(
            str)

        # face recognition configs
        self.fFaceConfThreshold = self.config_file['Classification']['fFaceConfThreshold'].get(
            float)
        self.fFrontalFaceThreshold = \
            self.config_file['Classification']['fFrontalFaceThreshold'].get(
                float)
        self.sFaceClassifierModel = self.config_file['Classification']['sFaceClassifierModel'].get(
            str)
        self.sFacenetModelCheckpoint = \
            self.config_file['Classification']['sFacenetModelCheckpoint'].get(
                str)
        self.sFacenetModelProtobuf = \
            self.config_file['Classification']['sFacenetModelProtobuf'].get(
                str)
        self.iNumFramesToConfirmIdentity = \
            self.config_file['Classification']['iNumFramesToConfirmIdentity'].get(
                int)

        # Debug configs
        self.bDebugMode = self.config_file['Debug']['bDebugMode'].get(bool)
        self.bDispVideoOutput = self.config_file['Debug']['bDispVideoOutput'].get(
            bool)
        self.bWriteOutputVideo = self.config_file['Debug']['bWriteOutputVideo'].get(
            bool)
        self.sOutputVideoFile = self.config_file['Debug']['sOutputVideoFile'].get(
            str)
        self.bAnalyzeExecTime = self.config_file['Debug']['bAnalyzeExecTime'].get(
            bool)

        # Face Detection config
        self.enumFaceDetectThreshold = self.config_file['Detection']['enumFaceDetectThreshold'].get(
        )
        self.fScaleFactor = self.config_file['Detection']['fScaleFactor'].get(
            float)
        self.iFaceMinsize = self.config_file['Detection']['iFaceMinsize'].get(
            int)
        self.iFaceDetectionInterval = self.config_file['Detection']['iFaceDetectionInterval'].get(
            int)
        self.iFaceCropSize = self.config_file['Detection']['iFaceCropSize'].get(
            int)
        self.iFaceCropMargin = self.config_file['Detection']['iFaceCropMargin'].get(
            int)

        # display output
        self.bDispFaceBbox = self.config_file['DispOnFrame']['bDispFaceBbox'].get(
            bool)
        self.bDispImgResults = self.config_file['DispOnFrame']['bDispImgResults'].get(
            bool)
        self.bDispTracks = self.config_file['DispOnFrame']['bDispTracks'].get(
            bool)
        self.bDispFaceLandmark = self.config_file['DispOnFrame']['bDispFaceLandmark'].get(
            bool)
        self.bDispHumanBbox = self.config_file['DispOnFrame']['bDispHumanBbox'].get(
            bool)
        self.bDispAgeGender = self.config_file['DispOnFrame']['bDispAgeGender'].get(
            bool)

        # Node / location configs
        self.bCreateSubDirForOutput = self.config_file['Node']['bCreateSubDirForOutput'].get(
            bool)
        self.fGpuMemoryFraction = self.config_file['Node']['fGpuMemoryFraction'].get(
            float)
        self.sCameraID = self.config_file['Node']['sCameraID'].get(str)
        self.sLocation = self.config_file['Node']['sLocation'].get(str)
        self.sOutputDir = self.config_file['Node']['sOutputDir'].get(str)
        self.sConfigOutFile = self.config_file['Node']['sConfigOutFile'].get(
            str)
        self.slogDir = self.config_file['Node']['slogDir'].get(str)
        self.sLogFile = self.config_file['Node']['sLogFile'].get(str)
        self.sLogFileEntry = self.config_file['Node']['sLogFileEntry'].get(str)
        self.sLogFileExit = self.config_file['Node']['sLogFileExit'].get(str)
        self.sLogFileOther = self.config_file['Node']['sLogFileOther'].get(str)
        self.sCapturedFaceDir = self.config_file['Node']['sCapturedFaceDir'].get(
            str)
        # self.sSaveEntryFaceDir = self.config_file['Node']['sSaveEntryFaceDir'].get(str)
        # self.sSaveExitFaceDir = self.config_file['Node']['sSaveExitFaceDir'].get(str)
        self.bEnableAgeGenderModule = self.config_file['Node']['bEnableAgeGenderModule'].get(
            bool)
        self.bEnableTracking = self.config_file['Node']['bEnableTracking'].get(
            bool)
        self.bUseYoloV3Tiny = self.config_file['Node']['bUseYoloV3Tiny'].get(
            bool)
        self.iProcessingInterval = self.config_file['Node']['iProcessingInterval'].get(
            int)

        # Tracker configs
        # self.fIOU = self.config_file['Tracker']['fIOU'].get(float)
        # self.fScore = self.config_file['Tracker']['fScore'].get(float)
        self.iYoloImageHeight = self.config_file['Tracker']['iYoloImageHeight'].get(
            int)
        self.iYoloImageWidth = self.config_file['Tracker']['iYoloImageWidth'].get(
            int)
        self.sAnchorsPathTiny = self.config_file['Tracker']['sAnchorsPathTiny'].get(
            str)
        self.sAnchorsPath = self.config_file['Tracker']['sAnchorsPath'].get(
            str)
        self.sDeepSortModel = self.config_file['Tracker']['sDeepSortModel'].get(
            str)
        self.sHumanDetectionModelTiny = self.config_file['Tracker']['sHumanDetectionModelTiny'].get(
            str)
        self.sHumanDetectionModel = self.config_file['Tracker']['sHumanDetectionModel'].get(
            str)
        self.sYoloClassesPath = self.config_file['Tracker']['sYoloClassesPath'].as_filename(
        )

        # Other Modules config
        self.sServerCert = self.config_file['OtherModules']['sServerCert'].get(
            str)
        self.sHAPostUrl = self.config_file['OtherModules']['sHAPostUrl'].get(
            str)

    def __str__(self):
        return 'Config Manager !!!'

    def saveConfig(self):
        config_dump = self.config_file.dump()
        configOutputDir = os.path.join(self.APPDIR, 'output', 'config')

        if not os.path.exists(configOutputDir):
            os.makedirs(configOutputDir)
        filePath = os.path.join(configOutputDir, self.sConfigOutFile)
        outFile = open(filePath, "w")
        outFile.write(config_dump)
        outFile.close()

        return filePath
