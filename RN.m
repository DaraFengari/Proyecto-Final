% Cargar los datos desde el archivo zoo.csv
data = csvread('zoo.csv');
X = data(:, 1:16)';
y = data(:, 17)';

% Dividir los datos en conjuntos de entrenamiento y prueba
train_ratio = 0.8;
test_ratio = 0.2;
num_samples = size(X, 2);
num_train = round(train_ratio * num_samples);

% Indices para la división de datos
indices = randperm(num_samples);

X_train = X(:, indices(1:num_train));
y_train = y(indices(1:num_train));
X_test = X(:, indices(num_train+1:end));
y_test = y(indices(num_train+1:end));

% Crear la red neuronal multicapa
hidden_layer_size = 10000;
net = patternnet(hidden_layer_size);

% Configurar el conjunto de entrenamiento
net.divideFcn = 'divideind';  % Utilizar un conjunto de entrenamiento personalizado
net.divideParam.trainInd = 1:num_train;
net.divideParam.valInd = [];  % No utilizar conjunto de validación
net.divideParam.testInd = num_train+1:num_samples;

% Configurar hiperparámetros de entrenamiento
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.01;

% Entrenar la red neuronal
net = train(net, X_train, y_train);

% Realizar predicciones en el conjunto de prueba con la red neuronal
y_pred = net(X_test);

% Convertir las salidas continuas a etiquetas de clase redondeando
y_pred_class = round(y_pred);

% Calcular la matriz de confusión
confMat = confusionmat(y_test, y_pred_class);

% Calcular métricas
accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
precision = confMat(2,2) / sum(confMat(:,2)); % TP / (TP + FP)
sensitivity = confMat(2,2) / sum(confMat(2,:)); % TP / (TP + FN)
specificity = confMat(1,1) / sum(confMat(1,:)); % TN / (TN + FP)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);

% Mostrar resultados
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Sensitivity: %.2f\n', sensitivity);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1_score);