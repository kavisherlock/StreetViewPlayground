#!/usr/bin/python

# Ref:https://github.com/sarahrn/Py-Gsvhn-DigitStruct-Reader

import sys
from helpers import digit_struct


def writeToCsvFile(dsFileName, dsCsvFile):
    fileCount = 0
    csvFile = open(dsCsvFile, 'w')
    csvFile.write("FileName,DigitLabel,Left,Top,Width,Height\n")
    for dsObj in digit_struct.yieldNextDigitStruct(dsFileName):
        fileCount += 1
        for bbox in dsObj.bboxList:
            csvLine = "{},{},{},{},{},{}\n".format(
                dsObj.name,
                bbox.label, bbox.left, bbox.top, bbox.width, bbox.height)
            csvFile.write(csvLine)
    print("Number of image files: {}".format(fileCount))
    csvFile.close()


def convertToCsv(dsFilePath, csvFilePath):
    with open(csvFilePath, "w") as csvFile:
        writeToCsvFile(dsFilePath, csvFile)


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("    {0} <inputMatFilePath> <outputCsvFilePath>".format(sys.argv[0]))
        return
    dsFilePath = sys.argv[1]
    csvFilePath = sys.argv[2]
    print("Converting {0} to {1}".format(dsFilePath, csvFilePath))
    convertToCsv(dsFilePath, csvFilePath)


if __name__ == "__main__":
    main()