# ============================================================================= #
#       Rede Neural básica baseada em perceptrons para classificação de dados.  #
#       Por Gabriel Ferreira.                                                   #
# ----------------------------------------------------------------------------- #
#       O objetivo desse código é demonstrar de maneira simples a construção    #
#   de uma rede neural baseada em perceptrons de múltipla camada com o Flux.    #
#                                                                               #
#       Para mais informações sobre o código acesse os arquivos relacionados    #
#   no GitHub:
# ============================================================================= #
#
#   Bibliotecas utilizadas no projeto
using Flux, Plots, Statistics, ProgressMeter
#   Tema do Plots para os gráficos
theme(:wong2)

# ============================================================================= #
#       * Função reponsável por tratar a saída da rede.                         #
# ----------------------------------------------------------------------------- #
#       Como a saída desejada é um valor de 1 a 10 que classifique qual das     #
#   10 velocidades angulares deu origem as características de entrada, pego     #
#   a maior probabilidade de saída na matriz de saída da rede e retorno um      #
#   valor associado classificando a linha da matriz.                            #
# ============================================================================= #
function get_graphdata_from_output(output)
    result = []
    for i = 1:100
        maior = 1
        if (output[2, i] > output[maior, i]) maior = 2 end
        if (output[3, i] > output[maior, i]) maior = 3 end
        if (output[4, i] > output[maior, i]) maior = 4 end
        if (output[5, i] > output[maior, i]) maior = 5 end
        if (output[6, i] > output[maior, i]) maior = 6 end 
        if (output[7, i] > output[maior, i]) maior = 7 end
        if (output[8, i] > output[maior, i]) maior = 8 end
        if (output[9, i] > output[maior, i]) maior = 9 end
        if (output[10, i] > output[maior, i]) maior = 10 end
        push!(result, maior)
    end
    return result
end
# ============================================================================= #
#       * Função responsável por gerar a matriz de dados desejados.             #
# ----------------------------------------------------------------------------- #
#       Essa função vai criar uma matriz de dimensão igual a saída da rede,     #
#   os valores desse matriz são basicamente preenchidos por zero, sendo o único #
#   valor diferente de zero igual a um e presente na linha referente ao         #
#   resultado esperado da saída, assim indicamos que a probabilidade desejada   #
#   é de 0% para todos menos o desejado, o qual é 100%.                         #
#       Uma dica é imprimir o resultado da rede e essa matrix para fazer uma    #
#   comparação dos valores de saída e do resultado desejado em cada posição.    #
# ============================================================================= #
function get_desired_values()
    desired_values = Matrix{Float32}(undef, 10, 100)
    for j = 1:100
        for i = 1:10
            desired_values[i, j] = 0
        end
        if j <= 10
            desired_values[1, j] = 1
            continue
        end
        if j <= 20
            desired_values[2, j] = 1
            continue
        end
        if j <= 30
            desired_values[3, j] = 1
            continue
        end
        if j <= 40
            desired_values[4, j] = 1
            continue
        end
        if j <= 50
            desired_values[5, j] = 1
            continue
        end
        if j <= 60
            desired_values[6, j] = 1
            continue
        end
        if j <= 70
            desired_values[7, j] = 1
            continue
        end
        if j <= 80
            desired_values[8, j] = 1
            continue
        end
        if j <= 90
            desired_values[9, j] = 1
            continue
        end
        if j <= 100
            desired_values[10, j] = 1
            continue
        end
    end
    return desired_values
end

# ============================================================================= #
#       * Construção dos dados que serão usados na rede.                        #
# ============================================================================= #
#
#   Listas de constantes que serão utilizadas para gerar os dados da rede.
#   ** Apensa os valores gerador por randn estão presentes na 𝜙_list.
𝜔_list = [0.411, 5.566, 1.624, 0.213, 0.074, 3.561, 1.222, 2.667, 0.316, 0.251]
𝜙_list = [2.108, 3.051, 0.438, 2.812, 0.810, 0.579, -2.830, -4.701, 0.865, 0.357]

#   Cria um conjunto de 1000 valores entre 0 e 10 que serão usados como entrada para
#   gerar as características da nossa função seno.
times = range(0, stop=10, length=1000)

#   Cria a matriz com os dados que serão utilizados no treinamento e nos testes.
#   ** O ideal é que esses dados sejam distintos, aqui no caso usarei o mesmo conjunto,
#   mas é preferível evitar isso.
data = Matrix{Float32}(undef, 100, 1000)
for i = 1:10
    for j = 1:10
        for p = 1:1000
            𝜔 = 𝜔_list[i]
            𝜙 = 𝜙_list[j]
            local t = times[p]
            data[(i - 1) * 10 + j, p] = sin(𝜔 * t + 𝜙)
        end
    end
end

#   Cria um conjunto de 100 valores de 0 a 100, variando de 1 em 1.
x = []
for i = 1:100
    push!(x, i)
end

#   Modelo da nossa rede neural.
model = Chain(
    Dense(1000, 500, relu),
    Dense(500, 125, relu),
    Dense(125, 50, relu),
    Dense(50, 10),
    softmax
)

#   Insere os dados na rede e gera um gráfico.
out_1 = get_graphdata_from_output(model(data'|>gpu)|>cpu)
graph_1 = scatter(x[1:100], out_1[1:100], legend=false)

#   Cria a matriz de valores desejados e carrega os dados com auxílio do Flux.
desired_values = get_desired_values()
loader = Flux.DataLoader((data', desired_values), batchsize = 64, shuffle = true) |> gpu

#   Determina a função de otimização e a função de perda.
optim = Flux.setup(Adam(0.01), model)
loss(y_hat,y) = Flux.crossentropy(y_hat, y)

#   Realiza o treinamento da rede.
losses = []
@showprogress for epochs = 1:375
    for (x, y) in loader
        loss_r, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            loss(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss_r)
    end
end

#   Insere novamente os dados na rede e gera um novo gráfico, agora com ela treinada.
out_2 = get_graphdata_from_output(model(data'|>gpu)|>cpu)
graph_2 = scatter(x[1:100], out_2[1:100], legend=false)

#   Gera um gráfico com as perdas registradas durante o treinamento.
graph_losse = plot(losses; xaxis=(:log10, "iteration"), yaxis="loss", label = "per batch")
n = length(loader)
plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)), label="epoch mean", dpi=400)

#   Imprime os gráficos gerados.
#   ** O graph_1 não está sendo impresso, use plot(graph_1, graph_2, graph_losse, layout=3)
#   caso deseje imprimir ele também.
plot(graph_2, graph_losse, layout=2)