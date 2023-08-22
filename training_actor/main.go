package main

import (
	"fmt"

	"federative-learning/messages"

	"io/ioutil"
	"net/http"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
	"github.com/golang/protobuf/proto"
)

type WeightsList struct {
	InnerWeightsLists [][]float64
}

type TrainingActor struct{}

func (state *TrainingActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.TrainRequest:
		client := &http.Client{}

		req, err := http.NewRequest("POST", "http://localhost:5000/train_nn", nil)
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
		fmt.Println("RESP BODY", resp.Body)
		// Read the response body
		weightsBytes, err := ioutil.ReadAll(resp.Body)

		if err != nil {
			fmt.Println("Error reading response body:", err)
			return
		}

		weightsList := &messages.WeightsList{}

		if err := proto.Unmarshal(weightsBytes, weightsList); err != nil {
			// Handle the error if unmarshaling fails
		}

		context.Send(msg.Sender, &messages.Response{
			Weights: weightsList,
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
