# Imports python modules
from os import listdir

def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    results_dic = {}
    filename_list = listdir(image_dir)
    
    for i in range(len(filename_list)):
       if filename_list[i][0] != "." and filename_list[i] not in results_dic:
          words_list = filename_list[i].lower().split('_')
          pet_label = ''

          for word in words_list:
              if word.isalpha():
                  pet_label += word + ' '
                  
          pet_label = pet_label.strip()
          results_dic[filename_list[i]] = [pet_label]
    



    return results_dic
