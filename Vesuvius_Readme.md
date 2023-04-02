# Vesuvius

Detect ink in 3d scans of scrolls buried by the eruption that that covered Pompeii and Herculaneum

volume-cartographer is how the virtual unwrapping has been done. [Link](https://github.com/educelab/volume-cartographer)

Turn it into a binary problem (ink vs no ink) for simplification, but isn't actually binary in real life

Full scroll could be a 4 class problem per voxel

1. Papyrus with no Ink
2. Papyrus with Ink
3. Empty Space
4. Carbon/Shadows/Burn Marks

As of 3/30
`0.12` is a silver submission at this time.
`0.48` is their sample submission to "beat" from InkId referenced below

## The Data

Data is stored in .tiff file with 8 micron resolution. It is a horizontal slice.

A volume is a 3D picture made up of 3D pixel cubes called voxels

Each scroll is ~14_375 tiff files and each Tiff file is 122 Mb!

Images are 560x560x1 pixels. (Single color image. reduces tensor depth)
16 bit precision on the individual number.

Campfire scrolls are 104µm for a tutorial @ 8bits

In here there are three folders:
    - raw: the raw X-ray photos of the scroll.
    - rec: the reconstructed 3D image volume (“rec” = “reconstructed”).
    - logs: log files during scanning and reconstruction.

Files
- [train/test]/[fragment_id]/surface_volume/[image_id].tif slices from the 3d x-ray surface volume. Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us width * height * 65 number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- [train/test]/[fragment_id]/mask.png — a binary mask of which pixels contain data.
- train/[fragment_id]/inklabels.png — a binary mask of the ink vs no-ink labels.
- train/[fragment_id]/inklabels_rle.csv — a run-length-encoded version of the labels, generated using this script. This is the same format as you should make your submission in.
- train/[fragment_id]/ir.png — the infrared photo on which the binary mask is based.
- sample_submission.csv, an example of a submission file in the correct format. You need to output the following file in the home directory: submission.csv. See the evaluation page for information.

They use `ImageJ-win64.exe` to view images.

### Run Length Encoding (RLE)

What is dis?

### Download the Data

[Download instructions](https://gist.github.com/nat/e7266a5c765686b7976df10d3a85041b)
Data Files are found [here](dl.ash2txt.org) registeredusers:only

Here is a command to download 1cm of scan data from the center of Scroll 1:

```bash
for i in `seq 6000 7250`; do wget --user=registeredusers --password=only http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volumes/20230205180739/0$i.tif; done
```

Had to specify my wget location as powershell aliases to their own command

```powershell
foreach($i in 6000..7250){wget --user=registeredusers --password=only http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volumes/20230205180739/0$i.tif;}
```

Total dataset is 8 TB?!?!? They have 180TB lying in wait
Resolution is 4 and 8 microns from a synchrotron.

Even the Fragments that were peeled off are actually multiple layers of papyri

The full scans are a couple hundred gbs each!

## Their Process

Acquisition
- Digitize the physical object
Segmentation
- Start with full CT scan of fragment surface. (Flatish)
- Capture the shape of the layers
- Find layers to build the pages.
- Results in a 3D mesh (.obj file) called a segment
Texturing
- Identify and amplify ink locations on pages
Flattening
- Flatten image
- Make twisted 3D pages easier to read and assemble
Alignment & Merging
- Generate a texture image using volume cartographer. Flattened image in x-ray
- Align the infrared image on top of the xraay and that becomes the "label" (Multimodal Image Registration)

Their `InkID` sample solution is a pretty basic CNN. It is based off of a geometric software pipeline. Each step in their pipeline has room for improvement.

This model's code should be available on their [github](https://github.com/educelab/ink-id) page.

## TODO

- [ ] Do EDA and plots to understand any correlation
- [ ] Wrap up the fragments tightly as they would have been in a scroll. Choose different diameters for it.
- [ ] Hu-Po's `eda.ipynb` file is a combination of example submission and the tutorial
- [ ] 

## Five People Question

If you had 5 clones of yourself, what instruction would you give each of them?

1. Get on kaggle and maximize the ink detection under its existing construction
2. Come up with methods to do segmentation of the big scrolls
3. More intelligent methods to do segmentation on the entire scrolls
4. Domain Transfer. Convert Image from fragments to look like it had come from within the big scrolls
5. Shake up the pipeline with new ideas. Not the existing geometric pipeline they use
6. More UI tools for manipulating the data (ImageJ)

# Papyrus

Papyrus is a grassy reed that grows along the banks of the Nile in Egypt. It can grow up to 4.5 meters tall and 7.5cm thick. The tough outer rind is peeled away. The green inner pith is peeled or sliced into strips.

The strips are laid out in two layers in a grid pattern. They are pressed together until the layers merge like velcro. And then left out in the sun to dry, where they turn light brown. 
The sheets – called kollemata – are smoothed out with an ivory or seashell ruler. 
The kollemata are then glued together with paste made of flour and water. Then the areas where they are joined are hammered smooth. This forms a long piece of papyrus, usually 15-30 feet, comprised of up to 20 kollemata.

The Papyrus is rolled up around a dowel called an umbilicus. Portions of it are unrolled for writing. The first section, called the protokollon, is usually left blank. Text is written in columns with a quill and inkwell. Inks are made of varied substances.