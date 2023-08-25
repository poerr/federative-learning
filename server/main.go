package main

import (
	"federative-learning/messages"
	"fmt"
	"time"

	"encoding/json"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
)

type (
	Initializer struct {
		pid            *actor.PID
		coordinatorPID *actor.PID
		loggerPID      *actor.PID
		aggregatorPID  *actor.PID
	}
	Coordinator struct {
		pid           *actor.PID
		parentPID     *actor.PID
		loggerPID     *actor.PID
		aggregatorPID *actor.PID
		remote_addr   string
		remote_port   uint
		remote_name   string

		roundsTrained  uint
		maxRounds      uint
		actorsTraining uint

		children *actor.PIDSet
		weights  []WeightsDictionary
	}
	Logger struct {
		pid       *actor.PID
		parentPID *actor.PID
	}
	Aggregator struct {
		pid       *actor.PID
		parentPID *actor.PID
	}
	pidsDtos struct {
		initPID        *actor.PID
		coordinatorPID *actor.PID
		loggerPID      *actor.PID
		aggregatorPID  *actor.PID
	}
	startTraining struct {
		trainingActorPID *actor.PID
		weights          [][]float32
	}
	spawnedRemoteActor struct {
		remoteActorPID *actor.PID
	}
	calculateAverage struct {
		weights []WeightsDictionary
	}
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

func newInitializatorActor() actor.Actor {
	return &Initializer{}
}

func newCoordinatorActor() actor.Actor {
	return &Coordinator{}
}

func newAggregatorActor() actor.Actor {
	return &Aggregator{}
}

func newLoggerActor() actor.Actor {
	return &Logger{}
}

func (state *Initializer) Receive(context actor.Context) {
	switch msg := context.Message().(type) {

	case *actor.PID:
		coorditatorProps := actor.PropsFromProducer(newCoordinatorActor)
		coorditatorPID := context.Spawn(coorditatorProps)
		aggregatorProps := actor.PropsFromProducer(newAggregatorActor)
		aggregatorPID := context.Spawn(aggregatorProps)
		loggerProps := actor.PropsFromProducer(newLoggerActor)
		loggerPID := context.Spawn(loggerProps)

		state.pid = msg
		state.aggregatorPID = aggregatorPID
		state.loggerPID = loggerPID
		state.coordinatorPID = coorditatorPID

		context.Send(loggerPID, pidsDtos{initPID: msg, loggerPID: loggerPID})
		context.Send(coorditatorPID, pidsDtos{initPID: msg, coordinatorPID: coorditatorPID, loggerPID: loggerPID, aggregatorPID: aggregatorPID})
		context.Send(aggregatorPID, pidsDtos{initPID: msg, aggregatorPID: aggregatorPID})

	case spawnedRemoteActor:
		// Passing the message to the cooridnator actor
		context.Send(state.coordinatorPID, msg)

	case startTraining:
		// Passing the message to the cooridnator actor
		context.Send(state.coordinatorPID, msg)
	}

}

func (state *Coordinator) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *actor.Started:
		state.children = actor.NewPIDSet()

	case pidsDtos:
		if msg.initPID == nil {
			return
		}
		state.parentPID = msg.initPID
		state.pid = msg.coordinatorPID
		state.loggerPID = msg.loggerPID
		state.aggregatorPID = msg.aggregatorPID
		state.maxRounds = 10
		state.roundsTrained = 0
		state.actorsTraining = 0

	case spawnedRemoteActor:
		// When a remote actor is spawned, we get a message
		// and put the remote actor's PID in the list of training actors
		state.children.Add(msg.remoteActorPID)

	case startTraining:
		// If we have reached maximum rounds of training we exit
		if state.roundsTrained >= state.maxRounds {
			fmt.Printf("Reached a maximum of %v training rounds.", state.maxRounds)
			return
		}
		// Create a training reqeuest message which will be sent
		// to all the training actors
		trainMessage := &messages.TrainRequest{Sender: state.pid}
		// Send the message to all the training actors
		state.children.ForEach(func(i int, pid *actor.PID) {
			context.Send(pid, trainMessage)
			state.actorsTraining += 1
		})

		state.roundsTrained += 1
		fmt.Println("TRAINING ROUND: ", state.roundsTrained)

	case *messages.TrainResponse:
		// Another node finished training
		state.actorsTraining -= 1
		// Convert the weights and add it to the weight list of this round of training
		var deserializedWeights WeightsDictionary
		json.Unmarshal(msg.Data, &deserializedWeights)

		weightsToAvg := append(state.weights, deserializedWeights)
		state.weights = weightsToAvg

		// All nodes finished training
		// send the weights to the aggregator
		if state.actorsTraining == 0 {
			context.Send(state.aggregatorPID, calculateAverage{weights: weightsToAvg})
		}
	}
}

func (state *Logger) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case pidsDtos:
		if msg.initPID == nil {
			return
		}
		state.parentPID = msg.initPID
		state.pid = msg.loggerPID

	}
}

func (state *Aggregator) Receive(context actor.Context) {

	switch msg := context.Message().(type) {
	case pidsDtos:
		if msg.initPID == nil {
			return
		}
		state.parentPID = msg.initPID
		state.pid = msg.aggregatorPID

	case calculateAverage:
		go FedAVG(msg.weights)
		context.Send(state.parentPID, startTraining{})
	}
}

func FedAVG(array []WeightsDictionary) {
	// Arrays to store averaged biases
	l1b_calculated := make([]float64, 64)
	l2b_calculated := make([]float64, 128)
	l3b_calculated := make([]float64, 128)
	l4b_calculated := make([]float64, 1)
	// Arrays to store averaged weights
	var l1w_calculated [24][64]float64
	var l2w_calculated [64][128]float64
	var l3w_calculated [128][128]float64
	var l4w_calculated [128][1]float64

	// Array of 2D weight arrays that need to be averaged
	var l1w_array [][24][64]float64
	var l2w_array [][64][128]float64
	var l3w_array [][128][128]float64
	var l4w_array [][128][1]float64

	// Array of arrays of biases that need to be averaged
	var l1b_array [][]float64
	var l2b_array [][]float64
	var l3b_array [][]float64
	var l4b_array [][]float64

	// Create a collection of all the arrays of the same sort
	for _, value := range array {
		l1w_array = append(l1w_array, value.Layer1_weights)
		l2w_array = append(l2w_array, value.Layer2_weights)
		l3w_array = append(l3w_array, value.Layer3_weights)
		l4w_array = append(l4w_array, value.Layer4_weights)

		l1b_array = append(l1b_array, value.Layer1_biases)
		l2b_array = append(l2b_array, value.Layer2_biases)
		l3b_array = append(l3b_array, value.Layer3_biases)
		l4b_array = append(l4b_array, value.Layer4_biases)
	}

	// Averaging the biases
	go calcAvg1D(l1b_array, l1b_calculated)
	go calcAvg1D(l2b_array, l2b_calculated)
	go calcAvg1D(l3b_array, l3b_calculated)
	go calcAvg1D(l4b_array, l4b_calculated)

	// Averaging the weights
	go calcAvgL1W(l1w_array, l1w_calculated)
	go calcAvgL2W(l2w_array, l2w_calculated)
	go calcAvgL3W(l3w_array, l3w_calculated)
	go calcAvgL4W(l4w_array, l4w_calculated)

	globalWeightsModel.Layer1_biases = l1b_calculated
	globalWeightsModel.Layer2_biases = l2b_calculated
	globalWeightsModel.Layer3_biases = l3b_calculated
	globalWeightsModel.Layer4_biases = l4b_calculated

	globalWeightsModel.Layer1_weights = l1w_calculated
	globalWeightsModel.Layer2_weights = l2w_calculated
	globalWeightsModel.Layer3_weights = l3w_calculated
	globalWeightsModel.Layer4_weights = l4w_calculated
}

// Averaging functions
func calcAvgL1W(array [][24][64]float64, result [24][64]float64) {

	length := float64(len(array))

	for _, array2d := range array {
		for i, array1d := range array2d {
			for j, value := range array1d {
				result[i][j] += value
			}
		}
	}
	for i, row := range result {
		for j := range row {
			result[i][j] /= length
		}
	}
}
func calcAvgL2W(array [][64][128]float64, result [64][128]float64) {

	length := float64(len(array))

	for _, array2d := range array {
		for i, array1d := range array2d {
			for j, value := range array1d {
				result[i][j] += value
			}
		}
	}
	for i, row := range result {
		for j := range row {
			result[i][j] /= length
		}
	}
}
func calcAvgL3W(array [][128][128]float64, result [128][128]float64) {

	length := float64(len(array))

	for _, array2d := range array {
		for i, array1d := range array2d {
			for j, value := range array1d {
				result[i][j] += value
			}
		}
	}
	for i, row := range result {
		for j := range row {
			result[i][j] /= length
		}
	}
}
func calcAvgL4W(array [][128][1]float64, result [128][1]float64) {

	length := float64(len(array))

	for _, array2d := range array {
		for i, array1d := range array2d {
			for j, value := range array1d {
				result[i][j] += value
			}
		}
	}
	for i, row := range result {
		for j := range row {
			result[i][j] /= length
		}
	}
}
func calcAvg1D(array [][]float64, result []float64) {

	length := float64(len(array))
	for _, row := range array {
		for i, value := range row {
			result[i] += value
		}
	}

	for i := range result {
		result[i] /= length
	}
}

// Global weights model
var globalWeightsModel WeightsDictionary

func main() {

	system := actor.NewActorSystem()
	decider := func(reason interface{}) actor.Directive {
		fmt.Println("handling failure for child")
		return actor.RestartDirective
	}

	supervisor := actor.NewOneForOneStrategy(10, 1000, decider)
	rootContext := system.Root
	props := actor.
		PropsFromProducer(newInitializatorActor,
			actor.WithSupervisor(supervisor))
	pid := rootContext.Spawn(props)

	rootContext.Send(pid, pid)

	remoteConfig := remote.Configure("127.0.0.1", 8000)
	remoting := remote.NewRemote(system, remoteConfig)
	remoting.Start()

	spawnResponse, err := remoting.SpawnNamed("127.0.0.1:8091", "training_actor", "training_actor", time.Second)

	if err != nil {
		panic(err)
	}

	spawnedActorMessage := spawnedRemoteActor{remoteActorPID: spawnResponse.Pid}
	rootContext.Send(pid, spawnedActorMessage)
	rootContext.Send(pid, startTraining{})

	_, _ = console.ReadLine()
}
