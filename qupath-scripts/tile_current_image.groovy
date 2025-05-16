// Modified version of https://github.com/andreped/NoCodeSeg/blob/9a16804126ab8d7efc217f196144f37d8fd6bd83/source/exportTiles.groovy

import qupath.lib.images.servers.LabeledImageServer
import java.awt.image.Raster
import javax.imageio.ImageIO;

// --------------------------------

def classNames = ["Tumor"]
def imageExtension = ".jpg"

def prefix = "wsi_tiled"
double downsample = 3
int patchSize = 512
int pixelOverlap = 256
def onlyAnnotated = true
def partialTiles = false
def customPath = // path to save tiles in

// --------------------------------

def imageData = getCurrentImageData()
def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
print "Current image: ${name}"

def fName = name.split("\\.")[0]

def fOnlyAnnotated = onlyAnnotated ? "T" : "F"
def fPartialTiles = partialTiles ? "T" : "F"

def folderName = "${prefix}|${fName}|${downsample}|${patchSize}|${pixelOverlap / patchSize}|${fOnlyAnnotated}|${fPartialTiles}"
def pathOutput = buildFilePath(customPath, folderName)
mkdirs(pathOutput)

getPathClass("Tumor").setColor(255, 255, 255)

def tempServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK)
    .downsample(downsample)
    .multichannelOutput(false)

def counter = 1
classNames.each { currClassName ->
    tempServer.addLabel(currClassName, counter)
    counter++;
}

def labelServer = tempServer.build()

new TileExporter(imageData)
    .downsample(downsample)
    .imageExtension(imageExtension)
    .labeledImageExtension(".png")
    .tileSize(patchSize)
    .labeledServer(labelServer)
    .labeledImageSubDir("masks")
    .imageSubDir("images")
    .annotatedTilesOnly(onlyAnnotated)
    .overlap(pixelOverlap)
    .includePartialTiles(partialTiles)
    .writeTiles(pathOutput)

Thread.sleep(100);
javafx.application.Platform.runLater {
    getCurrentViewer().getImageRegionStore().cache.clear();
    System.gc();
}

Thread.sleep(100);

print "Done!"
