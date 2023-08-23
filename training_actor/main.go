package main

import (
	"fmt"

	"federative-learning/messages"

	"encoding/json"
	"net/http"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
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
		// Read the response body
		// weightsBytes, err := io.ReadAll(resp.Body)
		// weightsBytesString := string(weightsBytes)
		// fmt.Println(weightsBytes)
		// if err != nil {
		// 	fmt.Println("Error reading response body:", err)
		// 	return
		// }

		weightsList := &messages.WeightsList{}

		var j interface{}
		decodingErr := json.NewDecoder(resp.Body).Decode(&j)
		if decodingErr != nil {
			fmt.Println(decodingErr)
			panic(decodingErr)
		}

		// if err := json.Unmarshal([]byte(weightsBytesString), &weightsList1); err != nil {
		// 	fmt.Println("GRESKA KOD MARSHALOVANJA: ", err)
		// }

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
