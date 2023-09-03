package main

import (
	"federative-learning/messages"
	"fmt"
	"log"
	"os"
	"reflect"
	"strconv"
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
	Aggregator struct {
		pid       *actor.PID
		parentPID *actor.PID
	}
	Coordinator struct {
		pid           *actor.PID
		parentPID     *actor.PID
		loggerPID     *actor.PID
		aggregatorPID *actor.PID

		roundsTrained  uint
		maxRounds      uint
		actorsTraining uint

		children *actor.PIDSet
		weights  []WeightsDictionary

		behavior actor.Behavior
	}
	Logger struct {
		pid       *actor.PID
		parentPID *actor.PID
	}

	LogMessage struct {
		Sender      *actor.PID
		Recipient   *actor.PID
		MessageType string
	}

	pidsDtos struct {
		initPID        *actor.PID
		coordinatorPID *actor.PID
		loggerPID      *actor.PID
		aggregatorPID  *actor.PID
	}

	startTraining    struct{}
	weightsUpdated   struct{}
	trainingFinished struct{}

	spawnedRemoteActor struct {
		remoteActorPID *actor.PID
	}
	calculateAverage struct {
		weights []WeightsDictionary
	}
	WeightsDictionary struct {
		Layer1_weights [24][64]float64
		Layer1_biases  [64]float64
		Layer2_weights [64][128]float64
		Layer2_biases  [128]float64
		Layer3_weights [128][128]float64
		Layer3_biases  [128]float64
		Layer4_weights [128][1]float64
		Layer4_biases  [1]float64
	}
)

func newInitializatorActor() actor.Actor {
	return &Initializer{}
}
func newAggregatorActor() actor.Actor {
	return &Aggregator{}
}
func newLoggerActor() actor.Actor {
	return &Logger{}
}
func NewSetCoordinatorBehavior() actor.Actor {
	act := &Coordinator{
		behavior: actor.NewBehavior(),
	}
	act.behavior.Become(act.Training)
	return act
}

func (state *Initializer) Receive(context actor.Context) {
	switch msg := context.Message().(type) {

	case *actor.PID:
		coorditatorProps := actor.PropsFromProducer(NewSetCoordinatorBehavior) // Setting the state to Training
		coorditatorPID := context.Spawn(coorditatorProps)
		aggregatorProps := actor.PropsFromProducer(newAggregatorActor)
		aggregatorPID := context.Spawn(aggregatorProps)
		loggerProps := actor.PropsFromProducer(newLoggerActor)
		loggerPID := context.Spawn(loggerProps)

		state.pid = msg
		state.aggregatorPID = aggregatorPID
		state.loggerPID = loggerPID
		state.coordinatorPID = coorditatorPID
		senderPID := context.Self()

		msg1 := pidsDtos{initPID: msg, loggerPID: loggerPID}
		context.Send(loggerPID, msg1)
		msgType1 := reflect.TypeOf(msg1).String()
		message1 := LogMessage{Sender: senderPID, Recipient: state.loggerPID, MessageType: msgType1}
		context.Send(state.loggerPID, message1)

		msg2 := pidsDtos{initPID: msg, coordinatorPID: coorditatorPID, loggerPID: loggerPID, aggregatorPID: aggregatorPID}
		context.Send(coorditatorPID, msg2)
		msgType2 := reflect.TypeOf(msg2).String()
		message2 := LogMessage{Sender: senderPID, Recipient: state.coordinatorPID, MessageType: msgType2}
		context.Send(state.loggerPID, message2)

		msg3 := pidsDtos{initPID: msg, aggregatorPID: aggregatorPID}
		context.Send(aggregatorPID, msg3)
		msgType3 := reflect.TypeOf(msg3).String()
		message3 := LogMessage{Sender: senderPID, Recipient: state.aggregatorPID, MessageType: msgType3}
		context.Send(state.loggerPID, message3)

		fmt.Printf("Initialized all actors %v\n", time.Now())

	case spawnedRemoteActor:
		// Passing the message to the cooridnator actor
		fmt.Printf("Spawned remote actor %v at %v\n", msg.remoteActorPID, time.Now())
		context.Send(state.coordinatorPID, msg)
		senderPID := context.Self()
		msgType4 := reflect.TypeOf(msg).String()
		message4 := LogMessage{Sender: senderPID, Recipient: state.coordinatorPID, MessageType: msgType4}
		context.Send(state.loggerPID, message4)

	case startTraining:
		// Passing the message to the cooridnator actor
		context.Send(state.coordinatorPID, msg)
		senderPID := context.Self()
		msgType5 := reflect.TypeOf(msg).String()
		message5 := LogMessage{Sender: senderPID, Recipient: state.coordinatorPID, MessageType: msgType5}
		context.Send(state.loggerPID, message5)
	case weightsUpdated:
		// Once the weights are updated, let coordinator know
		context.Send(state.coordinatorPID, msg)
		senderPID := context.Self()
		msgType6 := reflect.TypeOf(msg).String()
		message6 := LogMessage{Sender: senderPID, Recipient: state.coordinatorPID, MessageType: msgType6}
		context.Send(state.loggerPID, message6)
	case trainingFinished:
		// Once the training is finished we can serialize
		// the new weights
		file, err := os.Create("weightModel.json")

		if err != nil {
			panic(err)
		}

		defer file.Close()
		encoder := json.NewEncoder(file)

		encodingError := encoder.Encode(globalWeightsModel)
		if encodingError != nil {
			panic(err)
		}
		fmt.Println("Successfully written the weights to 'weightModel.json' file!")

	case LogMessage:
		context.Send(state.loggerPID, msg)
	}
}

func (state *Coordinator) Receive(context actor.Context) {
	state.behavior.Receive(context)
}

// This state signifies that the weights are ready and averaged
// so the training can start
func (state *Coordinator) Training(context actor.Context) {
	switch msg := context.Message().(type) {

	// When the actor is initialized, initialize an array of PIDs that
	// will represent all the training actors
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

	// When a remote actor is spawned, we get a message
	// and put the remote actor's PID in the list of training actors
	case spawnedRemoteActor:
		state.children.Add(msg.remoteActorPID)

	// When we receive this message, the coordinator sends messages to
	// all the training actors to start a new round of training
	case startTraining:
		state.weights = []WeightsDictionary{}
		fmt.Printf("Starting a new round of training at %v\n", time.Now())
		// If we have reached maximum rounds of training we exit
		if state.roundsTrained >= state.maxRounds {
			fmt.Printf("Reached a maximum of %v training rounds.\n", state.maxRounds)
			msg7 := trainingFinished{}
			context.Send(state.parentPID, msg7)
			senderPID := context.Self()
			msgType7 := reflect.TypeOf(msg7).String()
			message7 := LogMessage{Sender: senderPID, Recipient: state.parentPID, MessageType: msgType7}
			context.Send(state.loggerPID, message7)

			return
		}
		// Create a training reqeuest message which will be sent
		// to all the training actors
		weightsJson, marshalErr := json.Marshal(globalWeightsModel)

		if marshalErr != nil {
			panic(marshalErr)
		}

		senderAddress := localAddress + ":" + strconv.Itoa(port)
		trainMessage := &messages.TrainRequest{
			SenderAddress:    senderAddress,
			SenderId:         state.pid.Id,
			MarshaledWeights: weightsJson,
		}
		// Send the message to all the training actors
		state.children.ForEach(func(i int, pid *actor.PID) {
			msg8 := trainMessage
			senderPID := context.Self()
			context.Send(pid, msg8)
			msgType8 := reflect.TypeOf(msg8).String()
			message8 := LogMessage{Sender: senderPID, Recipient: pid, MessageType: msgType8}
			context.Send(state.loggerPID, message8)
			state.actorsTraining += 1
		})
		state.roundsTrained += 1
		fmt.Println("TRAINING ROUND: ", state.roundsTrained)

	case *messages.TrainResponse:
		fmt.Printf("Got a training response %v\n", time.Now())
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
			fmt.Printf("All actors finished training, the round %v has ended at %v\n", state.roundsTrained, time.Now())
			msg9 := calculateAverage{weights: weightsToAvg}
			senderPID := context.Self()
			context.Send(state.aggregatorPID, msg9)
			msgType9 := reflect.TypeOf(msg9).String()
			message9 := LogMessage{Sender: senderPID, Recipient: state.aggregatorPID, MessageType: msgType9}
			context.Send(state.loggerPID, message9)
			state.behavior.Become(state.WeightCalculating)
		}
	}
}

// This state signifies that one round of training is over
// so we have to wait until the weights are averaged and updated
// to start a new training round
func (state *Coordinator) WeightCalculating(context actor.Context) {

	switch msg := context.Message().(type) {
	case startTraining:
		fmt.Println("Waiting for new weight model!")
		fmt.Printf("msg: %v\n", msg)
	case weightsUpdated:
		state.behavior.Become(state.Training)
		msg10 := startTraining{}
		context.Send(state.parentPID, msg10)
		senderPID := context.Self()
		msgType10 := reflect.TypeOf(msg10).String()
		message10 := LogMessage{Sender: senderPID, Recipient: state.parentPID, MessageType: msgType10}
		context.Send(state.loggerPID, message10)
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
	case LogMessage:
		log.Printf("Sender: %v, Receiver: %v, Message type: %v", msg.Sender, msg.Recipient, msg.MessageType)
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

	// Once all the training actors have finished
	// we send all the weights to the aggregator
	// that calculates the average
	// Once it is done it tell the coordinator that it has finished
	// so it can start another round of training
	case calculateAverage:
		fmt.Printf("Averaging the weights %v\n", time.Now())
		fmt.Println("Length of all weights array:", len(msg.weights))
		FedAVG(msg.weights)
		fmt.Printf("Weights have been averaged %v\n", time.Now())
		msg11 := weightsUpdated{}
		context.Send(state.parentPID, msg11)
		senderPID := context.Self()
		msgType11 := reflect.TypeOf(msg11).String()
		message11 := LogMessage{Sender: senderPID, Recipient: state.parentPID, MessageType: msgType11}
		context.Send(state.parentPID, message11)
	}
}

func FedAVG(array []WeightsDictionary) {
	// Arrays to store averaged biases
	var l1b_calculated [64]float64
	var l2b_calculated [128]float64
	var l3b_calculated [128]float64
	var l4b_calculated [1]float64
	// Array of 2D weight arrays that need to be averaged
	var l1w_array [][24][64]float64
	var l2w_array [][64][128]float64
	var l3w_array [][128][128]float64
	var l4w_array [][128][1]float64

	// Array of arrays of biases that need to be averaged
	var l1b_array [][64]float64
	var l2b_array [][128]float64
	var l3b_array [][128]float64
	var l4b_array [][1]float64

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
	go calcAvgL1B(l1b_array, l1b_calculated)
	go calcAvgL2B(l2b_array, l2b_calculated)
	go calcAvgL3B(l3b_array, l3b_calculated)
	go calcAvgL4B(l4b_array, l4b_calculated)
	globalWeightsModel.Layer1_biases = l1b_calculated
	globalWeightsModel.Layer2_biases = l2b_calculated
	globalWeightsModel.Layer3_biases = l3b_calculated
	globalWeightsModel.Layer4_biases = l4b_calculated

	// Averaging the weights
	globalWeightsModel.Layer1_weights = calcAvgL1W(l1w_array)
	globalWeightsModel.Layer2_weights = calcAvgL2W(l2w_array)
	globalWeightsModel.Layer3_weights = calcAvgL3W(l3w_array)
	globalWeightsModel.Layer4_weights = calcAvgL4W(l4w_array)
}

// Averaging functions
func calcAvgL1W(array [][24][64]float64) [24][64]float64 {

	length := float64(len(array))
	var result [24][64]float64

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
	return result
}
func calcAvgL2W(array [][64][128]float64) [64][128]float64 {

	length := float64(len(array))
	var result [64][128]float64

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
	return result
}
func calcAvgL3W(array [][128][128]float64) [128][128]float64 {

	length := float64(len(array))
	var result [128][128]float64

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
	return result
}
func calcAvgL4W(array [][128][1]float64) [128][1]float64 {

	length := float64(len(array))
	var result [128][1]float64
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
	return result
}
func calcAvgL1B(array [][64]float64, result [64]float64) {

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
func calcAvgL2B(array [][128]float64, result [128]float64) {

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
func calcAvgL3B(array [][128]float64, result [128]float64) {

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
func calcAvgL4B(array [][1]float64, result [1]float64) {

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

// IP address and the port of the local actor system instance
var localAddress string
var port int

func main() {

	localAddress = "127.0.0.1"
	port = 8000

	file, openingErr := os.Open("weightModel.json")
	if openingErr != nil {
		panic(openingErr)
	}
	defer file.Close()

	logfile, _ := os.Create("log.txt")
	log.SetOutput(logfile)
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	defer logfile.Close()

	// Decode the JSON data from the file into a struct
	decoder := json.NewDecoder(file)
	deserializingError := decoder.Decode(&globalWeightsModel)
	if deserializingError != nil {
		panic(deserializingError)
	}

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

	msg12 := pid
	rootContext.Send(pid, msg12)
	senderPID := rootContext.Self()
	msgType12 := reflect.TypeOf(msg12).String()
	message12 := LogMessage{Sender: senderPID, Recipient: pid, MessageType: msgType12}
	rootContext.Send(pid, message12)

	remoteConfig := remote.Configure(localAddress, port)
	remoting := remote.NewRemote(system, remoteConfig)
	remoting.Start()

	spawnResponse, err1 := remoting.SpawnNamed("127.0.0.1:8091", "training_actor", "training_actor", time.Second)
	// spawnResponse1, err2 := remoting.SpawnNamed("192.168.0.24:8091", "training_actor", "training_actor", time.Second)

	if err1 != nil {
		panic(err1)
	}
	// if err2 != nil {
	// 	panic(err2)
	// }
	spawnedActorMessage := spawnedRemoteActor{remoteActorPID: spawnResponse.Pid}
	// spawnedActorMessage1 := spawnedRemoteActor{remoteActorPID: spawnResponse1.Pid}
	msg13 := spawnedActorMessage
	rootContext.Send(pid, msg13)
	msgType13 := reflect.TypeOf(msg13).String()
	message13 := LogMessage{Sender: senderPID, Recipient: pid, MessageType: msgType13}
	rootContext.Send(pid, message13)
	// rootContext.Send(pid, spawnedActorMessage1)
	msg14 := startTraining{}
	rootContext.Send(pid, msg14)
	msgType14 := reflect.TypeOf(msg14).String()
	message14 := LogMessage{Sender: senderPID, Recipient: pid, MessageType: msgType14}
	rootContext.Send(pid, message14)

	_, _ = console.ReadLine()
}
