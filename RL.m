% Cargar los datos desde el archivo CSV
data = readtable('zoo.csv'); 

% Separar los datos en características (X) y etiquetas (y)
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

% Dividir los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
rng(42); % Para reproducibilidad
idx = randperm(size(data, 1));
train_size = round(0.8 * size(data, 1));

X_train = X(idx(1:train_size), :);
y_train = y(idx(1:train_size));

X_test = X(idx(train_size+1:end), :);
y_test = y(idx(train_size+1:end));

% Convertir las etiquetas reales a categóricas
y_test_categorical = categorical(y_test);

% Entrenar el modelo de regresión logística
mdl = mnrfit(X_train, categorical(y_train));

% Realizar predicciones en el conjunto de prueba
y_pred = mnrval(mdl, X_test);

% Convertir las predicciones a clases
y_pred_class = predictClass(y_pred);

% Evaluar la eficiencia del modelo
accuracy = sum(y_pred_class == y_test_categorical) / numel(y_test_categorical);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Calcular la matriz de confusión
confMat = confusionmat(y_test_categorical, y_pred_class);

% Calcular Precision, Sensitivity, Specificity y F1 Score
precision = confMat(2,2) / sum(confMat(:,2)); % TP / (TP + FP)
sensitivity = confMat(2,2) / sum(confMat(2,:)); % TP / (TP + FN)
specificity = confMat(1,1) / sum(confMat(1,:)); % TN / (TN + FP)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);

fprintf('Precision: %.2f\n', precision);
fprintf('Sensitivity: %.2f\n', sensitivity);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1_score);

% ...

% Función para convertir las probabilidades a clases
function predictedClass = predictClass(probabilities)
    [~, idx] = max(probabilities, [], 2);
    predictedClass = categorical(idx);
end