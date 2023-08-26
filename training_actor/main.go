package main

import (
	"bytes"
	"fmt"
	"io"

	"federative-learning/messages"

	"net/http"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
)

type (
	WeightsDictionary struct {
		Layer1_weights [24][64]float64
		Layer1_biases  []float64
		Layer2_weights [64][128]float64
		Layer2_biases  []float64
		Layer3_weights [128][128]float64
		Layer3_biases  []float64
		Layer4_weights [128][1]float64
		Layer4_biases  []float64
	}
)
type WeightsList struct {
	InnerWeightsLists [][]float64
}

type TrainedWeights struct {
	Value interface{}
}

type TrainingActor struct{}

func (state *TrainingActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.TrainRequest:
		client := &http.Client{}

		req, err := http.NewRequest("POST", "http://localhost:5000/train_nn", bytes.NewBuffer(msg.MarshaledWeights))
		if err != nil {
			fmt.Println("Error creating request:", err)
			return
		}

		// Set headers if needed
		req.Header.Set("Content-Type", "application/json")

		// Send the request
		resp, err := client.Do(req)
		if err != nil {
			fmt.Println("Error sending request:", err)
			return
		}
		defer resp.Body.Close()

		weightsBytes, err := io.ReadAll(resp.Body)

		if err != nil {
			panic(err)
		}

		var senderPID actor.PID
		senderPID.Address = msg.SenderAddress
		senderPID.Id = msg.SenderId

		context.Send(&senderPID, &messages.TrainResponse{
			Data: weightsBytes,
		})
	}
}

func main() {

	system := actor.NewActorSystem()
	remoteConfig := remote.Configure("127.0.0.1", 8091)
	remoting := remote.NewRemote(system, remoteConfig)
	remoting.Start()

	remoting.Register("training_actor", actor.PropsFromProducer(func() actor.Actor { return &TrainingActor{} }))
	console.ReadLine()
}
