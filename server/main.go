package main

import (
	"federative-learning/messages"
	"fmt"
	"time"

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
		weights  [][][]float32
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
		weights [][][]float32
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
		context.Send(state.coordinatorPID, msg)

	case startTraining:
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
		state.children.Add(msg.remoteActorPID)

	case startTraining:
		if state.roundsTrained >= state.maxRounds {
			return
		}
		trainMessage := &messages.TrainRequest{}
		state.children.ForEach(func(i int, pid *actor.PID) {
			context.Send(pid, trainMessage)
			state.actorsTraining += 1
		})
		state.roundsTrained += 1

	case *messages.Response:
		// another node finished training
		state.actorsTraining -= 1

		// convert the weights and add it to the weight list of this round of training
		convertedWeights := ConvertWeightsList(msg.Weights)
		state.weights = append(state.weights, convertedWeights)

		if state.actorsTraining == 0 {
			// All nodes finished training
			// send the weights to the aggregator
			context.Send(state.aggregatorPID, calculateAverage{weights: state.weights})
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
		// calculate the average
		go FedAVG(msg.weights, globalWeightModel)
		// send the message to coordinator to start a new training round
		message := startTraining{weights: globalWeightModel}
		context.Send(state.parentPID, message)
	}
}

func FedAVG(weightsToSum [][][]float32, globalWeights [][]float32) {

	if len(weightsToSum) == 0 {
		panic("No weights were provided!")
	}

	summedWeightsArray := make([][]float32, len(weightsToSum[0][0]))

	for _, nodeWeights := range weightsToSum {
		for j, nodeLayerWeights := range nodeWeights {
			for k, weight := range nodeLayerWeights {
				summedWeightsArray[j][k] += weight
			}
		}
	}

	averageWeights := make([][]float32, len(summedWeightsArray))
	for i, sumWeightArray := range summedWeightsArray {
		for j, singleWeight := range sumWeightArray {
			averageWeights[i][j] = singleWeight / float32(len(weightsToSum))
		}
	}
	globalWeights = averageWeights
}

func ConvertWeightsList(weightsList *messages.WeightsList) [][]float32 {
	var weightsArray [][]float32

	for _, innerList := range weightsList.InnerList {
		weightsArray = append(weightsArray, innerList.ValueList)
	}

	return weightsArray
}

var globalWeightModel [][]float32

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
	// DEBUGGING
	// system.EventStream.Subscribe(func(event interface{}) {
	// 	if deadLetter, ok := event.(*actor.DeadLetterEvent); ok {
	// 		fmt.Println("Dead letter sender:", deadLetter)
	// 	}
	// })

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
