import sys, getopt
import cv2
from matplotlib import pyplot as plt

from contours_processing import ordered_merged_contours
from cube import Cube
from image_processing import create_floor_grad, img_with_lines


def get_cubes(img):
    contours = ordered_merged_contours(img)
    cubes = []
    # assert len(contours) == 2, len(contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        img_contours = img.copy()

        x_base = max(x - 10, 0)
        y_base = max(y - 10, 0)
        x_last = min(x + w + 10, len(img[0]))
        y_last = min(y + h + 10, len(img))

        cube = Cube(img_contours[y_base:y_last, x_base:x_last],
                    y_base,
                    y_last,
                    x_base,
                    x_last, 10)
        cubes.append(cube)

    return cubes


def pretty_show(img, img_edged, img_durty_lines, img_lines, img_grad,
                img_background):
    plt.subplot(231), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(img_edged, cmap='gray')
    plt.title('Edged'), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(img_durty_lines, cmap='gray')
    plt.title('Dirty lines'), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(img_lines, cmap='gray')
    plt.title('Clear lines'), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(img_grad, cmap='gray')
    plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(img_background, cmap='gray')
    plt.title('Depth map'), plt.xticks([]), plt.yticks([])

    plt.show()

def help_print():
    print('main.py is python script for building depth map')
    print('-i <inputfile>')
    print('-o <outputfile>')
    print('-s show all steps')
    print('-h print with message')
    print('you can not specify output file. in this case output image will display')

if __name__ == "__main__":

    argv = sys.argv[1:]
    inputfile = ''
    outputfile = ''
    need_all_show = False
    try:
        opts, args = getopt.getopt(argv,"shi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            help_print()
            sys.exit(0)
        if opt == '-s':
            need_all_show = True
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    if inputfile == '':
        print('Error: you need specify input file')
        help_print()
        sys.exit(0)

    origin_img = cv2.imread(inputfile)
    if origin_img is None:
        print("Error: can't open image file")
        exit(0)

    cubes = get_cubes(origin_img)
    if len(cubes) == 0:
        print("Error: can't find cubes in image")
        exit(0)

    print("n_cubes",len(cubes))
    cube1 = cubes[0]

    img_backgrnd= create_floor_grad(cubes, len(origin_img), len(origin_img[0]), cube1.vps)

    if outputfile == '':
        plt.imshow(img_backgrnd, cmap='gray')
        plt.title('Depth map'), plt.xticks([]), plt.yticks([])
        plt.show()
    elif not outputfile is None:
        cv2.imwrite(outputfile, img_backgrnd)

    if need_all_show:
        pretty_show(origin_img, cube1.img_edged, img_with_lines(cube1.img, cube1.dirty_lines),
                    cube1.img_lines, cube1.img_grad_cntred, img_backgrnd)

