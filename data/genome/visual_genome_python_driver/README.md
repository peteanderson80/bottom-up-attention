# Visual Genome Python Driver
A python wrapper for the [Visual Genome API](https://visualgenome.org/api/v0/). Visit the website for a complete list of [object models](https://visualgenome.org/api/v0/api_object_model.html) and details about all [endpoints](https://visualgenome.org/api/v0/api_endpoint_reference.html). Look at our [demo](https://github.com/ranjaykrishna/visual_genome_python_driver/blob/master/region_visualization_demo.ipynb) to see how you can use the python driver to access all the Visual Genome data.

### 2 ways of accessing the data
There are 2 ways of accessing the visual genome data.

1. Use the API functions to access the data directly from our server. You will not need to keep any local data available.
2. Download all the data and use our local methods to parse and work with the visual genome data. 
... You can download the data either from the [Visual Genome website](https://visualgenome.org/api/v0/) or by using the download scripts in the [data directory](https://github.com/ranjaykrishna/visual_genome_python_driver/tree/master/src/data).

### The API Functions are listed below.

#### Get all Visual Genome image ids
All the data in Visual Genome must be accessed per image. Each image is identified by a unique id. So, the first step is to get the list of all image ids in the Visual Genome dataset.

```python
> from src import api
> ids = api.GetAllImageIds()
> print ids[0]
1
```

`ids` is a python array of integers where each integer is an image id.

#### Get a range of Visual Genome image ids
There are 108,249 images currently in the Visual Genome dataset. Instead of getting all the image ids, you might want to just get the ids of a few images. To get the ids of images 2000 to 2010, you can use the following code:

```python
> ids = api.GetImageIdsInRange(startIndex=2000, endIndex=2010)
> print ids
[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
```

#### Get image data
Now, let's get basic information about an image. Specifically, for a image id, we will extract the url of the image, it's width and height (dimensions). We will also collect it's COCO and Flickr ids from their respective datasets.

```python
> image = api.GetImageData(id=61512)
> print image
id: 61512, coco_id: 248774, flickr_id: 6273011878, width: 1024, url: https://cs.stanford.edu/people/rak248/VG_100K/61512.jpg
```

`GetImageData` returns an `Image` model that you can read about in [src/models.py](https://github.com/ranjaykrishna/visual_genome_python_driver/blob/master/src/models.py).

#### Get Region Descriptions for an image
Now, let's get some exciting data: dense captions of an image. In Visual Genome, these are called region descriptions. Each region description is a textual description of a particular region in the image. A region is defined by it's top left coordinates (x, y) and a width and height.

```python
# Let's get the regions for image with id=61512
> regions = api.GetRegionDescriptionsOfImage(id=61512)
> print regions[0]
id: 1, x: 511, y: 241, width: 206, height: 320, phrase: A brown, sleek horse with a bridle, image: 61512
```

`GetRegionDescriptionsOfImage` returns an array of `Region` objects which are defined in [src/models.py](https://github.com/ranjaykrishna/visual_genome_python_driver/blob/master/src/models.py).
Check out our [demo](https://github.com/ranjaykrishna/visual_genome_python_driver/blob/master/region_visualization_demo.ipynb) to see these regions get visualized.

#### Get Region Graph from Region.
Let's get the region graph of the Region we printed out above. Region Graphs are tiny scene graphs for a particular region of an image. It contains: objects, attributes and relationships. Objects are localized in the image with bounding boxes. Attributes modify the object while Relationships are interactions between pairs of objects. We will get the scene graph of an image and print out the objects, attributes and relationships.

```python
# Remember that the region desription is 'A brown, sleek horse with a bridle'.
> graph = api.GetSceneGraphOfImage()
> print graph.objects
[horse]
>
>
> print graph.attributes
[horse is brown]
>
>
print graph.relationships
[]
```

The region graph has one object: `horse` and one attribute `brown` to describe the `horse`. It has no relationships.


#### Get Scene Graph for an image
Now, let's get the entire scene graph of an image. Each scene graph has three components: objects, attributes and relationships. Objects are localized in the image with bounding boxes. Attributes modify the object while Relationships are interactions between pairs of objects. We will get the scene graph of an image and print out the objects, attributes and relationships.

```python
> # First, let's get the scene graph
> graph = GetSceneGraphOfImage()
> # Now let's print out the objects. We will only print out the names and not the bounding boxes to make it look clean.
> print graph.objects
[horse, grass, horse, bridle, truck, sign, gate, truck, tire, trough, window, door, building, halter, mane, mane, leaves, fence]
>
>
> # Now, let's print out the attributes
> print graph.attributes
[3015675: horse is brown, 3015676: horse is spotted, 3015677: horse is red, 3015678: horse is dark brown, 3015679: truck is red, 3015680: horse is brown, 3015681: truck is red, 3015682: sign is blue, 3015683: gate is red, 3015684: truck is white, 3015685: tire is blue, 3015686: gate is wooden, 3015687: horse is standing, 3015688: truck is red, 3015689: horse is brown and white, 3015690: building is tan, 3015691: halter is red, 3015692: horse is brown, 3015693: gate is wooden, 3015694: grass is grassy, 3015695: truck is red, 3015696: gate is orange, 3015697: halter is red, 3015698: tire is blue, 3015699: truck is white, 3015700: trough is white, 3015701: horse is brown and cream, 3015702: leaves is green, 3015703: grass is lush, 3015704: horse is enclosed, 3015705: horse is brown and white, 3015706: horse is chestnut, 3015707: gate is red, 3015708: leaves is green, 3015709: building is brick, 3015710: truck is large, 3015711: gate is red, 3015712: horse is chestnut colored, 3015713: fence is wooden]
>
>
> # Finally, let's print out the relationships
> print graph.relationships
[3199950: horse stands on top of grass, 3199951: horse is in grass, 3199952: horse is wearing bridle, 3199953: trough is for horse, 3199954: window is next to door, 3199955: building has door, 3199956: horse is nudging horse, 3199957: horse has mane, 3199958: horse has mane, 3199959: trough is for horse]
```

#### Get Question Answers for an image
Let's now get all the Question Answers for one image. Each Question Answer object contains the id of the question-answer pair, the id of image, the question and the answer string, as well as the list of question objects and answer objects identified and canonicalized in the qa pair. We will extract the QAs for image 61512 and show all attributes of one such QA.

```python
> # First extract the QAs for this image
> qas = api.GetQAofImage(id=61512)
>
> # First print out some core information of the QA
> print qas[0]
id: 991154, image: 61512, question: What color is the keyboard?, answer: Black.
>
> # Now let's print out the question objects of the QA
> print qas[0].q_objects
[]
``` 
`GetQAofImage` returns an array of `QA` objects which are defined in [src/models.py](https://github.com/ranjaykrishna/visual_genome_python_driver/blob/master/src/models.py). The attributes `q_objects` and `a_objects` are both an array of `QAObject`, which is also defined there.

#### Get all Questions Answers in the dataset
We also have a function that allows you to get all the 1.7 million QAs in the Visual Genome dataset. If you do not want to get all the data, you can also specify how many QAs you want the function to return using the parameter `qtotal`. So if `qtotal = 10`, you will get back 10 QAs.

```python
> # Let's get only 10 QAs and print out the first QA.
> qas = api.GetAllQAs(qtotal=10)
> print qas[0]
id: 133103, image: 1159944, question: What is tall with many windows?, answer: Buildings.
```

To get all the QAs, set qtotal to None.

#### Get one type of Questions Answers from  the entire dataset
You might be interested in only collecting `why` questions. To query for a particular type of question, set `qtype` to `what`, `who`, `why`, `where`, `when`, `how`.

```python
> # Let's get the first 10 why QAs and print the first one.
> qas = GetQAofType(qtotal=10)
> print qas[0]
id: 133089, image: 1159910, question: Why is the man cosplaying?, answer: For an event.
```

### The local functions are listed below.

#### Downloading the data.
```bash
> # Download all the image data.
> ./src/data/getImageData.sh
>
> # Download all the region descriptions.
> ./src/data/getRegionDescriptions.sh
>
> # Download all the question answers.
> ./src/data/getQuestionAnswers.sh
```


#### Get Scene Graphs for 200 images from local .json files

```python
> import src.local as vg
> 
> # Convert full .json files to image-specific .jsons, save these to 'data/by-id'.
> # These files will take up a total ~1.1G space on disk.
> vg.SaveSceneGraphsById(dataDir='data/', imageDataDir='data/by-id/')
> 
> # Load scene graphs in 'data/by-id', from index 0 to 200.
> # We'll only keep scene graphs with at least 1 relationship.
> scene_graphs = vg.GetSceneGraphs(startIndex=0, endIndex=-1, minRels=1,
>                                  dataDir='data/', imageDataDir='data/by-id/')
> 
> print len(scene_graphs)
149
> 
> print scene_graphs[0].objects
[clock, street, shade, man, sneakers, headlight, car, bike, bike, sign, building, ... , street, sidewalk, trees, car, work truck]
```

### License
MIT License copyright Ranjay Krishna

### Questions? Comments?
My hope is that the API and the python wrapper are so easy that you never have to ask questions. But if you have any question, you can contact me directly at ranjaykrishna at gmail or contact the project at stanfordvisualgenome @ gmail.

Follow us on Twitter:
- [@RanjayKrishna](https://twitter.com/RanjayKrishna)
- [@VisualGenome](https://twitter.com/visualgenome)

### Want to Help?
If you'd like to help, write example code, contribute patches, document methods, tweet about it. Your help is always appreciated!

