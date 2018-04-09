# -*- coding: utf-8 -*-
'''
Created on Sep 17, 2015

@author: sebastian
'''
import datetime
import glob
from mpop.projector import get_area_def
from mpop.satellites import GeostationaryFactory
import numpy
import os
from pyorbital.astronomy import sun_earth_distance_correction
import shutil
import subprocess
import tarfile

# Methods for loading raw MSG data in HRIT-format, cropping to the LCRS domain, calibration, etc.
# PathToRawMSGData: Path to TAR-File with raw MSG data in HRIT-Format
def loadCalibratedDataForDomain(xrit_dir,xRITDecompressToolPath,correct=False,areaBorders=(-802607,3577980,1498701,5108186),slices=[7,8],hrv_slices=[17,18,19,20,21,22,23,24],channels=['VIS006','VIS008','IR_016','IR_039','IR_087','IR_108','IR_120','IR_134'], delete_all_files_in_folder = False, simpleProfiler = None):
    if (simpleProfiler is not None):
        simpleProfiler.start("loadCalibratedDataForDomain")

    # Dekomprimierung der HRIT-Dateien:
    decompressHRITFiles(xrit_dir,xRITDecompressToolPath,slices=slices, channels=channels, hrv_slices=hrv_slices, simpleProfiler = simpleProfiler)

    # Einlesen der Szene (auf LCRS-Ausschnitt beschränkt):
    scene_data, time_slot, error = readMeteosatScene(xrit_dir, correct, areaBorders, channels=channels, simpleProfiler = simpleProfiler)

    # Löschen der temporären Dateien:
    if delete_all_files_in_folder:
        deleteAllFilesInFolder(xrit_dir)

    if (simpleProfiler is not None):
        simpleProfiler.stop("loadCalibratedDataForDomain")

    return scene_data, time_slot, error

# Entpacke best. Kanäle aus tar-Datei. Epi- und Prolog werden IMMER mit entpackt
# channels:    String-Liste aller zu entpackender Channels
# slices:      Int-Liste aller zu entpackender slices
# tarfilePath: Pfad zur tar-File die entpackt werden soll
# outputPath:  Ordner in dem die extrahierten Dateien abgelegt werden sollen
# Für channels erlaubte Werte:
# HRV, IR_016, IR_039, IR_087, IR_097, IR_108, IR_120, IR_134, VIS006, VIS008, WV_062, WV_073
def extractChannelsFromTarFile(tarFilePath,outputPath,channels=['VIS006','VIS008','IR_016','IR_039','WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134'],slices=range(1,9), hrv_slices=range(1,25)):
    tarFile = tarfile.open(name=tarFilePath, mode='r')
    members = tarFile.getnames()

    for member in members:
        for channel in channels:
            slices = slices
            if channel == 'HRV':
                slices = hrv_slices
                channel = 'HRV___'

            for slicee in slices:
                if (channel in member and '___-' + "%06d" % slicee + '___-' in member):
                    tarFile.extract(member, path=outputPath)
                if ('EPI' in member or 'PRO' in member):
                    tarFile.extract(member, path=outputPath)
    return

# entpacke HRIT-Dateien in einem Ordner
# folder:                 Ordner mit zu entpackenden HRIT-Files
# xRITDecompressToolPath: Pfad zum xRITDecompressTool
# channels:               Zu extrahierende Kanäle, default: alle bis auf 'HRV'
# slices:                 Zu extrahierende Slices, default: alle
def decompressHRITFiles(folder,xRITDecompressToolPath,channels=['VIS006','VIS008','IR_016','IR_039','WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134'],slices=range(1,9), hrv_slices=range(1,25), simpleProfiler = None):

    if (simpleProfiler is not None):
        simpleProfiler.start("decompressHRITFiles")

    auswahl = []
    dateiListe = glob.glob(folder + 'H-000-MSG*-C_')
    for datei in dateiListe:
        for channel in channels:
            slices = slices
            if channel == 'HRV':
                slices = hrv_slices
                channel = 'HRV___'

            for slicee in slices:
                if (channel in datei and '___-' + "%06d" % slicee + '___-' in datei): 
                    auswahl.append(os.path.abspath(datei))
    absolut_xRITDecompressToolPath = os.path.abspath(xRITDecompressToolPath)
    current_dir = os.getcwd()
    os.chdir(folder) # Wechsel in das Input-Verzeichnis damit decompressed Files auch dort gespeichert werden
    for filepath in auswahl:
        subprocess.call([absolut_xRITDecompressToolPath, filepath]) # stdout=subprocess.PIPE damit Console nicht beschrieben wird.
    os.chdir(current_dir) # switch back to prev dir

    if (simpleProfiler is not None):
        simpleProfiler.stop("decompressHRITFiles")

    return

# Diese Methode lädt die ENTPACKTEN und ENTKOMPRIMIERTEN HRIT-Dateien in "folder" und speist
# damit eine "compound satellite scene". Diese wird zurück gegeben.
# area_def: Gebiet das geladen werden soll (muss in "areas.def" definiert worden sein. UND muss proj=geos sein !!!!
# channels: Zu extrahierende Kanäle, default: alle bis auf 'HRV'
# areaBorders: (left,down,right,up) (Default: LCRSProducts-Domain)
def readMeteosatScene(folder,correct,areaBorders=(-802607,3577980,1498701,5108186),channels=['VIS006','VIS008','IR_016','IR_039','WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134'], simpleProfiler = None):

    if (simpleProfiler is not None):
        simpleProfiler.start("readMeteosatScene")

    scene_data = None
    time_slot = None
    error_message = None

    dateiListe = glob.glob(folder + 'H-000-MSG*-__')
    # Automatisches Rausfinden des Meteosat-Typs
    satType = '10' # entspr. MSG3
    
    if dateiListe[0][dateiListe[0].rfind("/")+10:dateiListe[0].rfind("/")+11] == '2':
        satType = '09'
    
    if dateiListe[0][dateiListe[0].rfind("/")+10:dateiListe[0].rfind("/")+11] == '1':
        satType = '08'
        
    # Automatisches Rausfinden des Datums der Szene:
    try:
        time_slot = datetime.datetime.strptime(dateiListe[0][-15:-3],"%Y%m%d%H%M")
    except:
        error_message =  'FEHLER: INKORREKTES DATUMS-FORMAT IN DATEI ' + dateiListe[0]
    
    # Testen, ob Prolog und Epilog-Datei vorhanden sind:
    epiAndProExist = False
    for datei in dateiListe:
        if '-EPI_' in datei and not epiAndProExist:
            for dat in dateiListe:
                if '-PRO_' in dat:
                    epiAndProExist = True
                    break
    if not epiAndProExist:
        error_message = 'KEINE PRO/EPILOG-DATEI !!! - '
    else:
        if error_message is None:
            scene_data = GeostationaryFactory.create_scene("meteosat", satType, "seviri", time_slot)
            scene_data.load(channels, area_extent=areaBorders)
            if correct:
                scene_data = correction_sed_coszen(scene_data,time_slot) # sun-earth distance correction & cosine of the solar zenith angle correction
                scene_data = correction_co2(scene_data)                            # co2-correction

    return scene_data, time_slot, error_message

# sun-earth distance correction & cosine of the solar zenith angle correction:
# Da das mipp-Paket nur "r=R*pi/I" rechnet, wird hier noch "... * d(t)^2 / cos(sunzen(time,location))" gerechnet.
# Ausserdem: /100 um auf gleichen Wertebereich, wie Meike zu kommen (Nicht in Prozent sondern von 0 bis 1)
# Dies betrifft nur die "warmen Kanäle" VIS006, VIS008 und IR016 (HRV eigentlich auch aber der wird hier momentan noch garnicht behandelt)
# correctionFactor für "warme" Kanäle:
def correction_sed_coszen(scene_data,time_slot):
    s_e_d_corr = sun_earth_distance_correction(time_slot) # Distance between sun and earth in AU
    scene_data["VIS006"].data = scene_data["VIS006"].sunzen_corr(time_slot).data * s_e_d_corr**2/100
    scene_data["VIS008"].data = scene_data["VIS008"].sunzen_corr(time_slot).data * s_e_d_corr**2/100 
    scene_data["IR_016"].data = scene_data["IR_016"].sunzen_corr(time_slot).data * s_e_d_corr**2/100
    return scene_data

# co2 correction uses the difference between 10.8 and 13.4 to correct 3.9
# more details can be found here: http://eumetrain.org/IntGuide/PowerPoints/Channels/conversion.ppt
def correction_co2(scene_data):
    deltaTCO2 = (scene_data["IR_108"].data - scene_data["IR_134"].data) / 4.
    Rcorr = numpy.power(scene_data["IR_108"].data, 4.) - numpy.power((scene_data["IR_108"].data - deltaTCO2),4.)
    scene_data["IR_039"].data = numpy.power((numpy.power(scene_data["IR_039"].data, 4.) + Rcorr), 0.25)
    # co2 correction is also integrated in mpop:
    #scene_data["IR_039"].data = scene_data.co2corr()
    return scene_data

# Methode zum löschen aller Dateien in einem Ordner:
def deleteAllFilesInFolder(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    return