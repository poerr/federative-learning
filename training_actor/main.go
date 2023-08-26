package main

import (
	"bytes"
	"fmt"
	"io"
	"time"

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
	TrainingActor struct {
		behavior actor.Behavior
	}
)

// Training actor will have 2 states: AbleToTrain & TrainingInProgress
func NewSetTrainingActorBehavior() actor.Actor {
	act := &TrainingActor{
		behavior: actor.NewBehavior(),
	}
	act.behavior.Become(act.AbleToTrain)
	return act
}
func (state *TrainingActor) Receive(context actor.Context) {
	state.behavior.Receive(context)
}

// If the actor is in 'AbleToTrain' state, once it gets the
// TrainRequest message, it will start training
func (state *TrainingActor) AbleToTrain(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.TrainRequest:
		fmt.Printf("Training actor %v started training at %v\n", context.Self(), time.Now())
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
		fmt.Printf("Training actor %v finished training at %v\n", context.Self(), time.Now())
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

// If the actor is in 'TrainingInProgress' state, it will not
// start training again
func (state *TrainingActor) TrainingInProgress(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.TrainRequest:
		fmt.Println("Training already in progress!")
		fmt.Printf("Received message from %v", msg.SenderAddress+" "+msg.SenderId)
	}
}

func main() {

	system := actor.NewActorSystem()
	remoteConfig := remote.Configure("127.0.0.1", 8091)
	remoting := remote.NewRemote(system, remoteConfig)
	remoting.Start()

	remoting.Register("training_actor", actor.PropsFromProducer(NewSetTrainingActorBehavior))
	console.ReadLine()
}
