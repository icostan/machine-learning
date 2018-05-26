base_model = VGG19(weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)
