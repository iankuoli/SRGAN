from keras.utils.vis_utils import plot_model

iamges_path = ""

srgan_network = SRGANNetwork(img_width=32, img_height=32, batch_size=1)
srgan_model = srgan_network.build_srgan_pretrain_model()

# Plot the model
plot_model(srgan_model, to_file='SRGAN.png', show_shapes=True)

srgan_network = SRGANNetwork(img_width=32, img_height=32, batch_size=1)
srgan_network.pre_train_srgan(iamges_path, nb_epochs=1, nb_images=50000)