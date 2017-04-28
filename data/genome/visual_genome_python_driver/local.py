from models import Image, Object, Attribute, Relationship
from models import Region, Graph, QA, QAObject, Synset
import httplib
import json
import utils
import os, gc

"""
Get Image ids from startIndex to endIndex.
"""
def GetAllImageData(dataDir=None):
  if dataDir is None:
    dataDir = utils.GetDataDir()
  dataFile = os.path.join(dataDir, 'image_data.json')
  data = json.load(open(dataFile))
  return [utils.ParseImageData(image) for image in data]

"""
Get all region descriptions.
"""
def GetAllRegionDescriptions(dataDir=None):
  if dataDir is None:
    dataDir = utils.GetDataDir()
  dataFile = os.path.join(dataDir, 'region_descriptions.json')
  imageData = GetAllImageData(dataDir)
  imageMap = {}
  for d in imageData:
    imageMap[d.id] = d
  images = json.load(open(dataFile))
  output = []
  for image in images:
    output.append(utils.ParseRegionDescriptions(image['regions'], imageMap[image['id']]))
  return output

"""
Get all question answers.
"""
def GetAllQAs(dataDir=None):
  if dataDir is None:
    dataDir = utils.GetDataDir()
  dataFile = os.path.join(dataDir, 'question_answers.json')
  imageData = GetAllImageData(dataDir)
  imageMap = {}
  for d in imageData:
    imageMap[d.id] = d
  images = json.load(open(dataFile))
  output = []
  for image in images:
    output.append(utils.ParseQA(image['qas'], imageMap))
  return output


# --------------------------------------------------------------------------------------------------
# GetSceneGraphs and sub-methods

"""
Load a single scene graph from a .json file.
"""
def GetSceneGraph(image_id, images='data/', imageDataDir='data/by-id/', synsetFile='data/synsets.json'):
  if type(images) is str:
    # Instead of a string, we can pass this dict as the argument `images`
    images = {img.id:img for img in GetAllImageData(images)}

  fname = str(image_id) + '.json'
  image = images[image_id]
  data = json.load(open(imageDataDir + fname, 'r'))

  scene_graph = ParseGraphLocal(data, image)
  scene_graph = InitSynsets(scene_graph, synsetFile)
  return scene_graph

"""
Get scene graphs given locally stored .json files; requires `SaveSceneGraphsById`.

 startIndex, endIndex : get scene graphs listed by image, from startIndex through endIndex
 dataDir : directory with `image_data.json` and `synsets.json`
 imageDataDir : directory of scene graph jsons saved by image id (see `SaveSceneGraphsById`)
 minRels, maxRels: only get scene graphs with at least / less than this number of relationships
"""
def GetSceneGraphs(startIndex=0, endIndex=-1,
                   dataDir='data/visualgenome/', imageDataDir='data/visualgenome/by-id/',
                   minRels=0, maxRels=100):
  images = {img.id:img for img in GetAllImageData(dataDir)}
  scene_graphs = []

  img_fnames = os.listdir(imageDataDir)
  if (endIndex < 1): endIndex = len(img_fnames)

  for fname in img_fnames[startIndex : endIndex]:
    image_id = int(fname.split('.')[0])
    scene_graph = GetSceneGraph(image_id, images, imageDataDir, dataDir+'synsets.json')
    n_rels = len(scene_graph.relationships)
    if (minRels <= n_rels <= maxRels):
      scene_graphs.append(scene_graph)

  return scene_graphs

"""
Use object ids as hashes to `src.models.Object` instances. If item not
  in table, create new `Object`. Used when building scene graphs from json.
"""
def MapObject(object_map, obj):

  oid = obj['object_id']
  obj['id'] = oid
  del obj['object_id']

  if oid in object_map:
    object_ = object_map[oid]

  else:
    if 'attributes' in obj:
      attrs = obj['attributes']
      del obj['attributes']
    else:
      attrs = []
    if 'w' in obj:
      obj['width'] = obj['w']
      obj['height'] = obj['h']
      del obj['w'], obj['h']

    object_ = Object(**obj)

    object_.attributes = attrs
    object_map[oid] = object_

  return object_map, object_


"""
Modified version of `utils.ParseGraph`.
"""
global count_skips
count_skips = [0,0]

def ParseGraphLocal(data, image, verbose=False):
  global count_skips
  objects = []
  object_map = {}
  relationships = []
  attributes = []

  for obj in data['objects']:
    object_map, o_ = MapObject(object_map, obj)
    objects.append(o_)
  for rel in data['relationships']:
    if rel['subject_id'] in object_map and rel['object_id'] in object_map:
      object_map, s = MapObject(object_map, {'object_id': rel['subject_id']})
      v = rel['predicate']
      object_map, o = MapObject(object_map, {'object_id': rel['object_id']})
      rid = rel['relationship_id']
      relationships.append(Relationship(rid, s, v, o, rel['synsets']))
    else:
      # Skip this relationship if we don't have the subject and object in 
      #   the object_map for this scene graph. Some data is missing in this way.
      count_skips[0] += 1
  if 'attributes' in data:
    for attr in data['attributes']:
      a = attr['attribute']
      if a['object_id'] in object_map:
        attributes.append(Attribute(attr['attribute_id'], a['object_id'], a['names'], a['synsets']))
      else:
        count_skips[1] += 1
  if verbose:
    print 'Skipped {} rels, {} attrs total'.format(*count_skips)
  return Graph(image, objects, relationships, attributes)

"""
Convert synsets in a scene graph from strings to Synset objects.
"""
def InitSynsets(scene_graph, synset_file):
  syn_data = json.load(open(synset_file, 'r'))
  syn_class = {s['synset_name'] : Synset(s['synset_name'], s['synset_definition']) for s in syn_data}

  for obj in scene_graph.objects:
    obj.synsets = [syn_class[sn] for sn in obj.synsets]
  for rel in scene_graph.relationships:
    rel.synset = [syn_class[sn] for sn in rel.synset]
  for attr in scene_graph.attributes:
    obj.synset = [syn_class[sn] for sn in attr.synset]

  return scene_graph  


# --------------------------------------------------------------------------------------------------
# This is a pre-processing step that only needs to be executed once. 
# You can download .jsons segmented with these methods from:
#     https://drive.google.com/file/d/0Bygumy5BKFtcQ1JrcFpyQWdaQWM

"""
Save a separate .json file for each image id in `imageDataDir`.

Notes
-----
- If we don't save .json's by id, `scene_graphs.json` is >6G in RAM
- Separated .json files are ~1.1G on disk
- Run `AddAttrsToSceneGraphs` before `ParseGraphLocal` will work
- Attributes are only present in objects, and do not have synset info

Each output .json has the following keys:
  - "id"
  - "objects"
  - "relationships"
"""
def SaveSceneGraphsById(dataDir='data/visualgenome/', imageDataDir='data/visualgenome/by-id/'):
  if not os.path.exists(imageDataDir): os.mkdir(imageDataDir)

  all_data = json.load(open(os.path.join(dataDir,'scene_graphs.json')))
  for sg_data in all_data:
    img_fname = str(sg_data['image_id']) + '.json'
    with open(os.path.join(imageDataDir, img_fname), 'w') as f:
      json.dump(sg_data, f)

  del all_data
  gc.collect()  # clear memory


"""
Add attributes to `scene_graph.json`, extracted from `attributes.json`.

This also adds a unique id to each attribute, and separates individual
attibutes for each object (these are grouped in `attributes.json`).
"""
def AddAttrsToSceneGraphs(dataDir='data/visualgenome/'):
  attr_data = json.load(open(os.path.join(dataDir, 'attributes.json')))
  with open(os.path.join(dataDir, 'scene_graphs.json')) as f:
    sg_dict = {sg['image_id']:sg for sg in json.load(f)}

  id_count = 0
  for img_attrs in attr_data:
    attrs = []
    for attribute in img_attrs['attributes']:
      a = img_attrs.copy(); del a['attributes']
      a['attribute']    = attribute
      a['attribute_id'] = id_count
      attrs.append(a)
      id_count += 1
    iid = img_attrs['image_id']
    sg_dict[iid]['attributes'] = attrs

  with open(os.path.join(dataDir, 'scene_graphs.json'), 'w') as f:
    json.dump(sg_dict.values(), f)
  del attr_data, sg_dict
  gc.collect()


# --------------------------------------------------------------------------------------------------
# For info on VRD dataset, see:  
#   http://cs.stanford.edu/people/ranjaykrishna/vrd/

def GetSceneGraphsVRD(json_file='data/vrd/json/test.json'):
  """
  Load VRD dataset into scene graph format.
  """
  scene_graphs = []
  with open(json_file,'r') as f:
    D = json.load(f)

  scene_graphs = [ParseGraphVRD(d) for d in D]
  return scene_graphs


def ParseGraphVRD(d):
  image = Image(d['photo_id'], d['filename'], d['width'], d['height'], '', '')

  id2obj = {}
  objs = []
  rels = []
  atrs = []

  for i,o in enumerate(d['objects']):
    b = o['bbox']
    obj = Object(i, b['x'], b['y'], b['w'], b['h'], o['names'], [])
    id2obj[i] = obj
    objs.append(obj)

    for j,a in enumerate(o['attributes']):
      atrs.append(Attribute(j, obj, a['attribute'], []))

  for i,r in enumerate(d['relationships']):
    s = id2obj[r['objects'][0]]
    o = id2obj[r['objects'][1]]
    v = r['relationship']
    rels.append(Relationship(i, s, v, o, []))

  return Graph(image, objs, rels, atrs)

