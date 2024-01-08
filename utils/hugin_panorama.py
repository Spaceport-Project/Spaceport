#!/usr/bin/env python
import os
import glob
import subprocess
import cv2
import errno

class HuginPanorama():
    def __init__(self, images):
   
 
     # Get the path to where input and output images for the panorama are stored
        self.images = images
        self.cwd = os.path.dirname(self.images[0])
 

        print(f'List of files: {self.images}')
    # def create_namedpipes(self, file_name):
    #     fifo_name = os.path.join("/tmp", "fifo_" + file_name)
    #     if os.path.exists(fifo_name):
    #         os.remove(fifo_name)
    #     os.mkfifo(fifo_name)
    #     return fifo_name

    # def load_image_to_pipe(self, file_name, fifo_name):
    #     data = cv2.imread(os.path.join(self.images,file_name))
    #     if data is None:
    #         print("ERROR image file didn't load or doesn't exist: %s" % file_name)
    #         exit(1)
        
    #     # try:
    #     #     with os.open(fifo_name, os.O_WRONLY | os.O_NONBLOCK) as fifo:
    #     #         # Encode the image as PNG format and write it to the named pipe
    #     #         fifo.write(cv2.imencode('.png', data)[1])
    #     # except OSError as e:
    #     #     if e.errno != errno.ENXIO:  # Ignore "No such device or address" error
    #     #         raise
    #     try:
    #         server_file = os.open(fifo_name, os.O_WRONLY | os.O_NONBLOCK)
    #     except OSError as exc:
    #         if exc.errno == errno.ENXIO:
    #             server_file = None
    #         else:
    #             raise
    #     if server_file is not None:
    #         print ("Writing line to FIFO...")
    #         try:
    #             os.write(server_file, cv2.imencode('.png', data)[1])

    #             # os.write(server_file, data)
    #             print ("Done.")
    #         except OSError as exc:
    #             if exc.errno == errno.EPIPE:
    #                 pass
    #             else:
    #                 raise
    #         os.close(server_file)
        
     
    

   


    def get_image_filenames(self):
        """ returns space seperated list of files in the images_path """
        print(" ".join(os.listdir(self.images)))
        # return " ".join(os.listdir(self.images_path))
        return os.listdir(self.images)

    def hugin_stitch(self):
        file_names = self.images
        format = f'{file_names[0].split("_")[-3]}_{file_names[0].split("_")[-2]}'
        format = format.split('/')[-1]
        # file_names.sort(key= lambda x: int(x.split('_')[0]))
        file_names = " ".join(file_names)
        # for file in files:
        #     self.load_image(os.path.join(self.images_path,file))

        # fifo_pipes = [self.create_namedpipes(file) for file in file_names]
        # [self.load_image_to_pipe(file_names[id], fifo_pipes[id]) for id, pipe in enumerate(fifo_pipes)]

        # if fifo_pipes == None or fifo_pipes == "":
        #     print("Did not find any images to stitch in: %s" % self.images_path)
        #     return
        # fifo_pipes_str = " ".join(fifo_pipes)
        # print("Stitching files: %s" % fifo_pipes_str)
        
        # create project file
        # cmd = ["cat"]  # 'cat' reads from the named pipe
        # cmd.extend(fifo_pipes_str.split())
        # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # # sp = self.bash_exec(f'pto_gen -p 0 -f 60 -o pano.pto {fifo_pipes_str}', deferred=True)
        # # [self.load_image_to_pipe(file_names[id], fifo_pipes[id]) for id, pipe in enumerate(fifo_pipes)]
        # for id, pipe in enumerate(fifo_pipes):
        #     self.load_image_to_pipe(file_names[id], fifo_pipes[id])
        # output, _ = process.communicate()
        self.bash_exec(f'pto_gen -p 0 -f 60 -o pano.pto {file_names}')
        # do cpfind
        self.bash_exec('cpfind -o pano.pto --multirow --celeste pano.pto')
        # do clean
        self.bash_exec('cpclean -o pano.pto pano.pto')
        # do vertical lines
        self.bash_exec('linefind -o pano.pto pano.pto')
        # do optimize locations
        self.bash_exec('autooptimiser -a -m -l -s -o pano.pto pano.pto')
        # calculate size
        self.bash_exec('pano_modify -p 1 --fov=360x115 --canvas=8000x4000  --ldr-file=png -o pano.pto pano.pto')
        # stitch
        self.bash_exec(f'hugin_executor --stitching --prefix=output_{format} pano.pto')
        # compress
        # self.bash_exec('convert output.tif output.png')
        
        # [os.remove(pipe) for pipe in fifo_pipes]

        # output_file = os.path.join(self.images, 'output.png')
        # if not os.path.isfile(output_file):
        #     print('Hugin failed to create a panorama.')
        #     return

        print(f'Panorama saved to: {self.cwd}/output_{format}.png')

        print('Finished.')


    def cleanup(self, delete_images=False):
        # Hugin project files and output images
        files_to_be_deleted = ['output.tif', 'pano.pto']
        
        # Optionally delete images
        if delete_images:
            image_types = ('*.png', '*.jpg', '*.gif')
            for file_type in image_types:
                path = os.path.join(self.images,file_type)
                files_to_be_deleted.extend(glob.glob(path))

        # Do file deletion
        for file in files_to_be_deleted:
            file_to_be_deleted = os.path.join(self.cwd, file)
            if os.path.isfile(file_to_be_deleted):
                os.remove(file_to_be_deleted)

    def do_stitch(self):
        print('Stitching panorama...')
        self.cleanup()
        self.hugin_stitch()

    def bash_exec(self, cmd):
        sp=subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cwd)
      
        out, err = sp.communicate()
        if out:
            print("\n%s" % out)
        if err:
            print("\n%s" % err)
        return out, err, sp.returncode
    

if __name__ == "__main__":
    pano = HuginPanorama("images_all")
    pano.do_stitch()