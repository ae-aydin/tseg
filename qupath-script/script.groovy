// Modified version of https://github.com/andreped/NoCodeSeg/blob/9a16804126ab8d7efc217f196144f37d8fd6bd83/source/exportTiles.groovy

import qupath.lib.images.servers.LabeledImageServer
import java.awt.image.Raster
import javax.imageio.ImageIO;

// --------------------------------

def classNames = ["Tumor"]
double downsample = 3  
int patchSize = 640  
int pixelOverlap = 160  
def imageExtension = ".jpg"
def multiChannel = false;
def onlyAnnotated = true;

// --------------------------------

def imageData = getCurrentImageData()

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, "tiles_${downsample}_${patchSize}_${pixelOverlap}_${onlyAnnotated}")
mkdirs(pathOutput)

def tempServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK)
    .downsample(downsample)
    .multichannelOutput(multiChannel)

def counter = 1
classNames.each { currClassName ->
    tempServer.addLabel(currClassName, counter)  
    counter++;
}

def labelServer = tempServer.build()

new TileExporter(imageData)
    .downsample(downsample)         
    .imageExtension(imageExtension)
    .tileSize(patchSize)
    .labeledServer(labelServer)
    .annotatedTilesOnly(onlyAnnotated)
    .overlap(pixelOverlap)
    .writeTiles(pathOutput)

print "Done!"

Thread.sleep(100);
javafx.application.Platform.runLater {
    getCurrentViewer().getImageRegionStore().cache.clear();
    System.gc();
}
Thread.sleep(100);